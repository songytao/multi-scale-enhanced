import argparse
import os
import random
import time

# PyTorch includes
import torch
import torch.optim as optim
# Tensorboard include
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
# Dataloaders includes
from dataloaders import tn3k, tg3k, tatn
from dataloaders import custom_transforms as trforms
from dataloaders import utils
# Model includes



from model.unet import Unet

from model.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from config import get_config

from model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from model.swintr_ca import swin_large_patch4_window12_384_in22k as swin_unet_large
# Loss function includes
from model.utils import soft_dice, soft_mse, boundary_loss



from thop import profile

import logging

# import cProfile
# import re
# cProfile.run('re.compile("ccc")', filename='result.out')



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    ## Model settings
    # unet, trfe, trfe1, trfe2, mtnet, segnet, deeplab-resnet50, fcn
    # TransUnet: ViT-B_16, ViT-B_32, ViT-L_16, ViT-L_32, ViT-H_14, R50-ViT-B_16, R50-ViT-L_16,SETR-UNT,ViT_seg,u2net, convformer,  egeunet,mf(MissFormer)
    parser.add_argument('-model_name', type=str, default='nst')
    parser.add_argument('-criterion', type=str, default='Dice')
    parser.add_argument('-pretrain', type=str, default= 'THYROI')  # THYROID

    parser.add_argument('-num_classes', type=int, default=1)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-output_stride', type=int, default=16)
    ## for transunet
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is 3')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')

    ## Train settings
    parser.add_argument('-dataset', type=str, default='TN3K')  # TN3K, TG3K, TATN, new data
    parser.add_argument('-dataname', type=str, default='tn3k')
    parser.add_argument('-fold', type=str, default=random.choice(['0','1','2','3','4']))
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-nepochs', type=int, default=400)
    parser.add_argument('-resume_epoch', type=int, default=0)

    ## Optimizer settings
    parser.add_argument('-naver_grad', type=str, default=1)
    parser.add_argument('-lr', type=float, default=1e-2)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-update_lr_every', type=int, default=10)
    parser.add_argument('-weight_decay', type=float, default=5e-4)

    ## Visualization settings
    parser.add_argument('-save_every', type=int, default=10)
    parser.add_argument('-log_every', type=int, default=50)
    parser.add_argument('-load_path', type=str, default='')
    parser.add_argument('-use_val', type=int, default=1)
    parser.add_argument('-use_test', type=int, default=1)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1234)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(save_dir_root, 'run', args.model_name, 'randomfold',args.dataname)
    log_dir = os.path.join(save_dir, 'log')
    writer = SummaryWriter(log_dir=log_dir)
    batch_size = args.batch_size
    config = get_config(args)
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3


    if 'unet' == args.model_name:
        net = Unet(in_ch=3, out_ch=args.num_classes)
    elif 'nst' == args.model_name:
        net = swin_unet_large(num_classes = args.num_classes,embed_dim=192,base_chanel= 64)

    else:
        raise NotImplementedError


    if args.resume_epoch == 0:
        print('Training ' + args.model_name + ' from scratch...')
    else:
        load_path = os.path.join(save_dir, args.model_name + '_epoch-' + str(args.resume_epoch) + '.pth')
        print('Initializing weights from: {}...'.format(load_path))
        net.load_state_dict(torch.load(load_path))

    if args.pretrain == 'THYROID':
        net.load_state_dict(torch.load('./model/transunet/imagenet21k_R50+ViT-B_16.npz', map_location=lambda storage, loc: storage),strict=False)
        print('loading pretrain model......')

    torch.cuda.set_device(device=0)
    net.cuda()

    print_params = 1

      # 定义好的网络模型
    if print_params == 1:
        input = torch.randn(1, 3, 224, 224)
        input = input.cuda()
        logging.getLogger('thop').setLevel(logging.WARNING)
        flops, params = profile(net, (input,))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1048576.0))

    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.criterion == 'Dice':
        criterion = soft_dice
    else:
        raise NotImplementedError

    composed_transforms_tr = transforms.Compose([
        trforms.FixedResize(size=(int(args.input_size), int(args.input_size)),num_classes = args.num_classes),
        trforms.RandomHorizontalFlip(num_classes = args.num_classes),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),num_classes = args.num_classes),
        trforms.ToTensor(num_classes = args.num_classes)])

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size),num_classes = args.num_classes),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),num_classes = args.num_classes),
        trforms.ToTensor(num_classes = args.num_classes)])

    if args.dataset == 'TN3K':
        args.fold = random.choice(['0','1','2','3','4'])
        train_data = tn3k.TN3K(mode='train', transform=composed_transforms_tr, fold=args.fold, dataname = args.dataname,num_classes = args.num_classes)
        val_data = tn3k.TN3K(mode='val', transform=composed_transforms_ts, fold=args.fold,dataname = args.dataname, num_classes = args.num_classes)
    elif args.dataset == 'TG3K':
        train_data = tg3k.TG3K(mode='train', transform=composed_transforms_tr)
        val_data = tg3k.TG3K(mode='val', transform=composed_transforms_ts)
    elif args.dataset == 'TATN':
        train_data = tatn.TATN(mode='train', transform=composed_transforms_tr, fold=args.fold)
        val_data = tatn.TATN(mode='val', transform=composed_transforms_ts, fold=args.fold)
    test_data = tn3k.TN3K(mode='test', transform=composed_transforms_ts, return_size=True, dataname = args.dataname, num_classes = args.num_classes)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                             pin_memory=True)
    valloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    num_iter_tr = len(trainloader)
    num_iter_va = len(valloader)
    num_iter_ts = len(testloader)
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = args.resume_epoch * len(train_data)
    print('nitrs: %d num_iter_tr: %d' % (nitrs, num_iter_tr))
    print('nsamples: %d tot_num_samples: %d' % (nsamples, len(train_data)))

    log_txt = open(log_dir + '/log.txt', 'w')
    aveGrad = 0
    global_step = 0
    best_flag = 1.0
    recent_losses = []
    nodule_losses = []
    thyroid_losses = []
    start_t = time.time()

    for epoch in range(args.resume_epoch, args.nepochs):
        if args.dataset == 'TN3K':
            args.fold = random.choice(['0', '1', '2', '3', '4'])
            train_data = tn3k.TN3K(mode='train', transform=composed_transforms_tr, fold=args.fold,dataname = args.dataname,num_classes = args.num_classes)
            val_data = tn3k.TN3K(mode='val', transform=composed_transforms_ts, fold=args.fold,dataname = args.dataname,num_classes = args.num_classes)
        elif args.dataset == 'TG3K':
            train_data = tg3k.TG3K(mode='train', transform=composed_transforms_tr)
            val_data = tg3k.TG3K(mode='val', transform=composed_transforms_ts)
        elif args.dataset == 'TATN':
            train_data = tatn.TATN(mode='train', transform=composed_transforms_tr, fold=args.fold)
            val_data = tatn.TATN(mode='val', transform=composed_transforms_ts, fold=args.fold)

        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                                 pin_memory=True)
        valloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        net.train()
        epoch_losses = []
        epoch_nodule_losses = []
        epoch_thyroid_losses = []
        epoch_mse_losses = []
        for ii, sample_batched in enumerate(trainloader):
            if 'trfe' in args.model_name or args.model_name == 'mtnet':
                nodules, glands = sample_batched
                scale = nodules['scale'].cuda()
                inputs_n, labels_n = nodules['image'].cuda(), nodules['label'].cuda()
                inputs_g, labels_g = glands['image'].cuda(), glands['label'].cuda()
                inputs = torch.cat([inputs_n[0].unsqueeze(0), inputs_g[0].unsqueeze(0)], dim=0)

                for i in range(1, inputs_n.size()[0]):
                    inputs = torch.cat([inputs, inputs_n[i].unsqueeze(0)], dim=0)
                    inputs = torch.cat([inputs, inputs_g[i].unsqueeze(0)], dim=0)

                global_step += inputs.data.shape[0]
                if 'trfeplus' in args.model_name:
                    nodule, thyroid, pred_scale = net.forward(inputs)
                    pred_scales = torch.zeros(int(len(pred_scale) / 2))
                    for i in range(len(pred_scales)):
                        pred_scales[i] = pred_scale[i * 2]
                else:
                    nodule, thyroid = net.forward(inputs)
                loss = 0
                nodule_loss_mini= 0
                thyroid_loss_mini = 0
                mse_loss_mini = 0
                for i in range(inputs.size()[0]):
                    if i % 2 == 0:
                        nodule_loss = criterion(nodule[i], labels_n[int(i / 2)])
                        nodule_loss_mini += nodule_loss
                    else:
                        thyroid_loss = 1 * criterion(thyroid[i], labels_g[int((i - 1) / 2)])
                        thyroid_loss_mini += thyroid_loss
                if 'trfeplus' in args.model_name:
                    mse_loss_mini = (1 - epoch / args.nepochs) * soft_mse(pred_scales.cuda(), scale.float())
                    loss += mse_loss_mini
                    log_txt.write(str(round(nodule_loss_mini.item(), 3)) + ' ' + str(
                        round(thyroid_loss_mini.item(), 3)) + ' ' + str(round(mse_loss_mini.item(), 3)) + '\n')
                else:
                    log_txt.write(
                        str(round(nodule_loss_mini.item(), 3)) + ' ' + str(round(thyroid_loss_mini.item(), 3)) + '\n')
                loss += (nodule_loss_mini + thyroid_loss_mini)
            elif args.model_name == 'nst':
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                global_step += inputs.data.shape[0]
                outputs,outputs1,outputs2,outputs3,outputs4 = net.forward(inputs)
                loss = criterion(outputs, labels)
                pool1 = torch.nn.MaxPool2d(2)
                labels1 = pool1(labels)
                #print(outputs1.shape,labels1.shape)
                #loss1 = criterion(outputs1, labels1)
                pool2 = torch.nn.MaxPool2d(2)
                labels2 = pool2(labels1)
               # loss2 = criterion(outputs2, labels2)
                pool3 = torch.nn.MaxPool2d(2)
                labels3 = pool3(labels2)
                #loss3 = criterion(outputs3, labels3)
                pool4 = torch.nn.MaxPool2d(2)
                labels4 = pool4(labels3)

                #loss4 = criterion(outputs4, labels4)

            elif args.model_name == 'egeunet':
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                global_step += inputs.data.shape[0]
                (outputs5, outputs4, outputs3, outputs2, outputs1) , outputs0 = net.forward(inputs)
                # print(outputs5.shape,outputs4.shape,outputs3.shape)
                loss = criterion(outputs0, labels)
                # pool1 = torch.nn.MaxPool2d(2)
                # labels1 = pool1(labels)
                loss1 = criterion(outputs1, labels)
                # pool2 = torch.nn.MaxPool2d(2)
                # labels2 = pool2(labels1)
                loss2 = criterion(outputs2, labels)
                # pool3 = torch.nn.MaxPool2d(2)
                # labels3 = pool3(labels2)
                loss3 = criterion(outputs3, labels)
                # pool4 = torch.nn.MaxPool2d(2)
                # labels4 = pool4(labels3)

                loss4 = criterion(outputs4, labels)
                # pool5 = torch.nn.MaxPool2d(2)
                # labels5 = pool5(labels4)
                loss5 = criterion(outputs5,labels)



                loss = (2/5)*loss +(1/5)*loss1+(4/25)*loss2+(3/25)*loss3+(2/25)*loss4+(1/25)*loss5

            else:
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                global_step += inputs.data.shape[0]
                outputs= net.forward(inputs)
                
                if args.model_name == 'u2net':
                    loss = criterion(outputs[0], labels)
                else:
                    # print("out:",outputs.shape,'lab: ',labels.shape)

                    loss = criterion(outputs, labels)


            if 'trfe' in args.model_name or args.model_name == 'mtnet':
                nodule_loss = nodule_loss.item()
                thyroid_loss = thyroid_loss.item()
                epoch_nodule_losses.append(nodule_loss_mini.item())
                epoch_thyroid_losses.append(thyroid_loss_mini.item())
                # epoch_mse_losses.append(mse_loss_mini.item())
                if len(nodule_losses) < args.log_every:
                    nodule_losses.append(nodule_loss)
                    thyroid_losses.append(thyroid_loss)
                else:
                    nodule_losses[nitrs % len(nodule_losses)] = nodule_loss
                    thyroid_losses[nitrs % len(thyroid_losses)] = thyroid_loss
            trainloss = loss.item()
            epoch_losses.append(trainloss)
            if len(recent_losses) < args.log_every:
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss

            # Backward the averaged gradient
            # loss = loss+loss1+loss2+loss3+loss4
            # loss = loss/5
            loss.backward()
            aveGrad += 1
            nitrs += 1
            nsamples += args.batch_size

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % args.naver_grad == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            if nitrs % args.log_every == 0:
                meanloss = sum(recent_losses) / len(recent_losses)
                print('epoch: %d ii: %d trainloss: %.2f timecost:%.2f secs' % (
                    epoch, ii, meanloss, time.time() - start_t))
                writer.add_scalar('data/trainloss', meanloss, nsamples)
                if 'trfe' in args.model_name or args.model_name == 'mtnet':
                    writer.add_scalar('data/train_nodule_loss', sum(nodule_losses) / len(nodule_losses), nsamples)
                    writer.add_scalar('data/train_thyroid_loss', sum(thyroid_losses) / len(thyroid_losses), nsamples)

        meanloss = sum(epoch_losses) / len(epoch_losses)
        print('epoch: %d meanloss: %.2f' % (epoch, meanloss))
        writer.add_scalar('data/epochloss', meanloss, nsamples)
        if 'trfe' in args.model_name or args.model_name == 'mtnet':
            writer.add_scalar('data/epoch_nodule_loss', sum(epoch_nodule_losses) / len(epoch_nodule_losses), nsamples)
            writer.add_scalar('data/epoch_thyroid_loss', sum(epoch_thyroid_losses) / len(epoch_thyroid_losses),
                              nsamples)

        if args.use_val == 1:
            prec_lists = []
            recall_lists = []
            sum_testloss = 0.0
            total_mae = 0.0
            count = 0
            iou = 0
            net.eval()
            for ii, sample_batched in enumerate(valloader):
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                with torch.no_grad():
                    if 'trfe' in args.model_name or args.model_name == 'mtnet':
                        if 'trfeplus' in args.model_name:
                            outputs, _, _ = net.forward(inputs)
                        else:
                            outputs, _ = net.forward(inputs)
                    elif 'cpfnet' in args.model_name:
                        main_out = net(inputs)
                        outputs = main_out[:, 0, :, :].view(1, 1, args.input_size, args.input_size)
                    elif 'nst' in args.model_name:
                        outputs = net.forward(inputs)[0]
                    elif args.model_name == 'egeunet':
                        _, outputs = net.forward(inputs)

                    else:
                        outputs = net.forward(inputs)

                    if 'v3' in args.model_name:
                        outputs = net(inputs)['out']
                        loss = criterion(outputs, labels)
                    elif 'cpfnet' in args.model_name:
                        loss = criterion(main_out, labels.long())
                    elif 'trfe' in args.model_name or args.model_name == 'mtnet':
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, labels)
                sum_testloss += loss.item()

                predictions = torch.sigmoid(outputs)

                iou += utils.get_iou(predictions, labels)
                count += 1

                total_mae += utils.get_mae(predictions, labels) * predictions.size(0)
                # print(predictions.size(),labels.size())
                prec_list, recall_list = utils.get_prec_recall(predictions, labels)
                prec_lists.extend(prec_list)
                recall_lists.extend(recall_list)

                if ii % num_iter_va == num_iter_va - 1:
                    mmae = total_mae / count
                    mean_testloss = sum_testloss / num_iter_va
                    mean_prec = sum(prec_lists) / len(prec_lists)
                    mean_recall = sum(recall_lists) / len(recall_lists)
                    fbeta = 1.3 * mean_prec * mean_recall / (0.3 * mean_prec + mean_recall)
                    iou = iou / count

                    print('Validation:')
                    print('epoch: %d, numImages: %d valloss: %.2f mmae: %.4f fbeta: %.4f iou: %.4f' % (
                        epoch, count, mean_testloss, mmae, fbeta, iou))
                    writer.add_scalar(f'data/{args.dataset}_validloss', mean_testloss, nsamples)
                    writer.add_scalar(f'data/{args.dataset}_validmae', mmae, nsamples)
                    writer.add_scalar(f'data/{args.dataset}_validfbeta', fbeta, nsamples)
                    writer.add_scalar(f'data/{args.dataset}_validiou', iou, epoch)

        if args.use_test == 1:
            prec_lists = []
            recall_lists = []
            sum_testloss = 0.0
            total_mae = 0.0
            count = 0
            iou = 0
            net.eval()
            for ii, sample_batched in enumerate(testloader):
                inputs, labels = sample_batched['image'].cuda(), sample_batched['label'].cuda()
                with torch.no_grad():
                    if 'trfe' in args.model_name or args.model_name == 'mtnet':
                        if 'trfeplus' in args.model_name:
                            outputs, _, _ = net.forward(inputs)
                        else:
                            outputs, _ = net.forward(inputs)
                    elif 'cpfnet' in args.model_name:
                        main_out = net(inputs)

                        outputs = main_out[:, 0, :, :].view(1, 1, args.input_size, args.input_size)
                    elif 'nst' in args.model_name:
                        outputs = net.forward(inputs)[0]

                    elif args.model_name == 'egeunet':
                        _,outputs = net.forward(inputs)
                    else:
                        outputs = net.forward(inputs)
                        # print('main_out:', outputs.shape)

                    if 'v3' in args.model_name:
                        outputs = net(inputs)['out']
                        loss = criterion(outputs, labels)
                    elif 'cpfnet' in args.model_name:
                        loss = criterion(main_out, labels.long())
                    elif 'trfe' in args.model_name or args.model_name == 'mtnet':
                        loss = criterion(outputs, labels)

                    else:
                        loss = criterion(outputs, labels)
                sum_testloss += loss.item()

                predictions = torch.sigmoid(outputs)

                iou += utils.get_iou(predictions, labels)
                count += 1

                total_mae += utils.get_mae(predictions, labels) * predictions.size(0)
                prec_list, recall_list = utils.get_prec_recall(predictions, labels)
                prec_lists.extend(prec_list)
                recall_lists.extend(recall_list)

                if ii % num_iter_ts == num_iter_ts - 1:
                    mmae = total_mae / count
                    mean_testloss = sum_testloss / num_iter_ts
                    mean_prec = sum(prec_lists) / len(prec_lists)
                    mean_recall = sum(recall_lists) / len(recall_lists)
                    fbeta = 1.3 * mean_prec * mean_recall / (0.3 * mean_prec + mean_recall)
                    iou = iou / count

                    print('test:')
                    print('epoch: %d, numImages: %d testloss: %.2f mmae: %.4f fbeta: %.4f iou: %.4f' % (
                        epoch, count, mean_testloss, mmae, fbeta, iou))
                    writer.add_scalar(f'data/{args.dataset}_validloss', mean_testloss, nsamples)
                    writer.add_scalar(f'data/{args.dataset}_validmae', mmae, nsamples)
                    writer.add_scalar(f'data/{args.dataset}_validfbeta', fbeta, nsamples)
                    writer.add_scalar(f'data/{args.dataset}_validiou', iou, epoch)
                    if mean_testloss <best_flag:
                        save_path = os.path.join(save_dir, args.model_name + 'NST_large+192+64+oct'+args.dataname+str(args.nepochs) + '.pth')
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        torch.save(net.state_dict(), save_path)
                        print("Save model at {}\n".format(save_path))
                        best_flag = mean_testloss

        # if epoch % args.save_every == args.save_every - 1:
        #     save_path = os.path.join(save_dir, args.model_name, f"fold{args.fold}", args.model_name + '_epoch-' + str(epoch) + '.pth')
        #     torch.save(net.state_dict(), save_path)
        #     print("Save model at {}\n".format(save_path))

        if epoch % args.update_lr_every == args.update_lr_every - 1:
            lr_ = utils.lr_poly(args.lr, epoch, args.nepochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            optimizer = optim.SGD(
                net.parameters(),
                lr=lr_,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
    writer.close()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
