import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders import custom_transforms as trforms
# Dataloaders includes
from dataloaders import tn3k, ddti
from dataloaders import utils
# Custom includes
from visualization.metrics import Metrics, evaluate


from model.unet import Unet

from model.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg



from model.swintr_ca import swin_large_patch4_window12_384_in22k as new_swin_unet_large

from config import get_config


from thop import profile



import shutil


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-model_name', type=str,default='nst')  # unet, mtnet, segnet, deeplab-resnet50, fcn, trfe, trfe1, trfe2,swinunet_tiny,trfeplus,unetplus
    parser.add_argument('-num_classes', type=int, default=2)
    parser.add_argument('-input_size', type=int, default=224)
    parser.add_argument('-output_stride', type=int, default=16)
    parser.add_argument('-load_path', type=str, default=r'./run/nst/randomfold/oct/nstNST_large+192+64+octoct400.pth')#run\nst\randomfold\nst+0.8loss+48+16+400+0.01.pth
    parser.add_argument('-save_dir', type=str, default='./results')
    parser.add_argument('-test_dataset', type=str, default='TN3K')
    parser.add_argument('-test_fold', type=str, default='test')
    parser.add_argument('-batch_size', type=int, default='1')
    parser.add_argument('-fold', type=int, default=0)
    parser.add_argument('-dataname', type=str, default='oct')    #  nt, nb , busi , # for transunet
    parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is 3')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = get_config(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(save_dir_root, 'run', args.model_name, 'randomfold',args.dataname)
    log_dir = os.path.join(save_dir, 'log')
    # writer = SummaryWriter(log_dir=log_dir)
    # batch_size = args.batch_size
    config = get_config(args)
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    # FIXME add other models

    if 'unet' == args.model_name:
        net = Unet(3, args.num_classes)
    elif 'nst' == args.model_name:
        net = new_swin_unet_large(num_classes = args.num_classes, embed_dim=192,base_chanel= 64)
    else:
        raise NotImplementedError
    # 参数测试
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(net, (input,))
    # flops = flops / 1e9  # 转换为 GFLOPs
    # params = (params * 4) / (1024 * 1024)  # 转换为 MB
    # print('flops: ', flops, 'params: ', params)



    net.load_state_dict(torch.load(args.load_path),strict=False)
    net.cuda()
    print_params = 1

      # 定义好的网络模型
    if print_params == 1:
        input = torch.randn(1, 3, 224, 224)
        input = input.cuda()
        flops, params = profile(net, (input,))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1048576.0))


    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size),num_classes = args.num_classes),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),num_classes = args.num_classes),
        trforms.ToTensor(num_classes = args.num_classes)])

    if args.test_dataset == 'TN3K':
        test_data = tn3k.TN3K(mode='test', transform=composed_transforms_ts, return_size=True,dataname=args.dataname,num_classes= args.num_classes)
    if args.test_dataset == 'DDTI':
        test_data = ddti.DDTI(transform=composed_transforms_ts, return_size=True)

    save_dir = args.save_dir + os.sep + args.test_fold + '-' + args.test_dataset + os.sep + args.model_name + os.sep + 'nst_large+192+64+octa' + os.sep
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    num_iter_ts = len(testloader)





    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net.cuda()
    net.eval()
    with torch.no_grad():
        if args.num_classes == 1:
            all_start = time.time()
            metrics = Metrics(['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])

            total_iou = 0
            total_cost_time = 0
            for sample_batched in tqdm(testloader):
                inputs, labels, label_name, size = sample_batched['image'], sample_batched['label'], sample_batched.get(
                    'label_name'), sample_batched['size']

                labels = labels.cuda()
                inputs = inputs.cuda()
                if 'trfe' in args.model_name or 'mtnet' in args.model_name:
                    if 'trfeplus' in args.model_name:
                        start = time.time()
                        nodule_pred, gland_pred, _ = net.forward(inputs)
                        cost_time = time.time() - start
                    else:
                        start = time.time()
                        nodule_pred, gland_pred = net.forward(inputs)
                        cost_time = time.time() - start
                    gland_pred = torch.sigmoid(gland_pred)
                elif 'cpfnet' in args.model_name:
                    start = time.time()
                    nodule_pred = net(inputs)
                    cost_time = time.time() - start
                else:
                    start = time.time()
                    nodule_pred = net.forward(inputs)[0]
                    cost_time = time.time() - start
                prob_pred = torch.sigmoid(nodule_pred)
                iou = utils.get_iou(prob_pred, labels)
                if prob_pred.shape != labels.shape :

                    prob_pred = prob_pred.unsqueeze(1)
                _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd = evaluate(prob_pred, labels)

                metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                               F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, hd=_hd, auc=_auc)
                save_flag = 0

                if save_flag == 1:
                    if _dice <=0.8085:
                        # 定义源文件和目标文件的路径
                        img_source = r"E:\swint_unet\data\tn3k\test-image"
                        img_destination = r"E:\swint_unet\data\busi\tn3k_less_avr_data\image"
                        mask_source = r'E:\swint_unet\data\tn3k\test-mask'
                        mask_destination = r'E:\swint_unet\data\busi\tn3k_less_avr_data\mask'


                        # 调用shutil.copy()函数复制文件
                        img_source = img_source+os.sep+label_name[0]
                        mask_source = mask_source + os.sep+label_name[0]

                        shutil.copy(img_source, img_destination)
                        shutil.copy(mask_source,mask_destination)


                total_iou += iou
                total_cost_time += cost_time

                shape = (size[0, 0], size[0, 1])
                prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data
                save_data = prob_pred[0]
                save_png = save_data[0].numpy()
                save_saliency = save_png * 255
                save_saliency = save_saliency.astype(np.uint8)

                save_png = np.round(save_png)
                # print(save_png.shape)

                save_png = save_png * 255
                save_png = save_png.astype(np.uint8)
                save_path = save_dir + label_name[0]
                if not os.path.exists(save_path[:save_path.rfind('/')]):
                    os.makedirs(save_path[:save_path.rfind('/')])
                save_path_s = save_dir + 's' + label_name[0]
                cv2.imwrite(save_path_s, save_saliency)
                cv2.imwrite(save_dir + label_name[0], save_png)
            print(args.model_name)
            metrics_result = metrics.mean(len(testloader))
            print("Test Result:")
            print(
                'recall: %.4f, specificity: %.4f, precision: %.4f, F1_score:%.4f, acc: %.4f, iou: %.4f, mae: %.4f, dice: %.4f, hd: %.4f, auc: %.4f'
                % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                   metrics_result['F1_score'],
                   metrics_result['acc'], metrics_result['iou'], metrics_result['mae'], metrics_result['dice'],
                   metrics_result['hd'], metrics_result['auc']))
            print("total_cost_time:", total_cost_time)
            print("loop_cost_time:", time.time() - all_start)
            evaluation_dir = os.path.sep.join([args.save_dir, 'metrics', args.test_fold + '-' + args.test_dataset + '/'])
            if not os.path.exists(evaluation_dir):
                os.makedirs(evaluation_dir)
    
            # keys_txt = ''
            metrics_result['inference_time'] = total_cost_time / len(testloader)
            values_txt = str(args.fold) + '\t'
            for k, v in metrics_result.items():
                if k != 'mae' or k != 'hd':
                    v = 100 * v
                # keys_txt += k + '\t'
                values_txt += '%.2f' % v + '\t'
            text = values_txt + '\n'
            save_path = evaluation_dir + args.model_name + args.dataname + '.txt'
            with open(save_path, 'a+') as f:
                f.write(text)
            print(f'metrics saved in {save_path}')
            print("------------------------------------------------------------------")

        else:
            all_start = time.time()
            metrics0 = Metrics(
                ['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])
            metrics1 = Metrics(
                ['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])

            total_iou = 0
            total_cost_time = 0
            for sample_batched in tqdm(testloader):
                inputs, labels, label_name, size = sample_batched['image'], sample_batched['label'], sample_batched.get(
                    'label_name'), sample_batched['size']

                labels = labels.cuda()
                inputs = inputs.cuda()
                if 'trfe' in args.model_name or 'mtnet' in args.model_name:
                    if 'trfeplus' in args.model_name:
                        start = time.time()
                        nodule_pred, gland_pred, _ = net.forward(inputs)
                        cost_time = time.time() - start
                    else:
                        start = time.time()
                        nodule_pred, gland_pred = net.forward(inputs)
                        cost_time = time.time() - start
                    gland_pred = torch.sigmoid(gland_pred)
                elif 'cpfnet' in args.model_name:
                    start = time.time()
                    nodule_pred = net(inputs)
                    cost_time = time.time() - start
                else:
                    start = time.time()
                    nodule_pred = net.forward(inputs)[0]
                    # print(nodule_pred.shape)
                    cost_time = time.time() - start
                prob_pred = torch.sigmoid(nodule_pred)
                iou = utils.get_iou(prob_pred, labels)
                if prob_pred.shape != labels.shape:
                    prob_pred = prob_pred.unsqueeze(0)
                # print(prob_pred.shape,labels.shape)
                prob_pred0 = prob_pred[:, 0, :, :]
                prob_pred1 = prob_pred[:,1, :, :]
                # Split labels into two tensors for two targets
                labels0 = labels[:, 0, :, :]
                labels1 = labels[:, 1, :, :]
                # print('p0,l0:',prob_pred0.shape,labels0.shape)
                _precision0, _recall0, _specificity0, _f10, _auc0, _acc0, _iou0, _dice0, _mae0, _hd0 = evaluate(prob_pred0, labels0)
                _precision1, _recall1, _specificity1, _f11, _auc1, _acc1, _iou1, _dice1, _mae1, _hd1 = evaluate(prob_pred1,
                                                                                                      labels1)


                metrics0.update(recall=_recall0*100, specificity=_specificity0*100, precision=_precision0*100,
                                F1_score=_f10*100, acc=_acc0*100, iou=_iou0*100, mae=_mae0*100, dice=_dice0*100, hd=_hd0*100, auc=_auc0*100)

                metrics1.update(recall=_recall1*100, specificity=_specificity1*100, precision=_precision1*100,
                                F1_score=_f11*100, acc=_acc1*100, iou=_iou1*100, mae=_mae1*100, dice=_dice1*100, hd=_hd1, auc=_auc1*100)




                total_iou += iou
                total_cost_time += cost_time

                shape = (size[0, 0], size[0, 1])
                prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data
                save_data = prob_pred[0]
                save_png0 = save_data[0].numpy()
                save_png1 = save_data[1].numpy()
                save_saliency0 = save_png0 * 255
                save_saliency1 = save_png1 * 255
                save_saliency0 = save_saliency0.astype(np.uint8)
                save_saliency1 = save_saliency1.astype(np.uint8)

                save_png0 = np.round(save_png0)
                save_png1 = np.round(save_png1)
                # print(save_png.shape)

                save_png0 = save_png0 * 255
                save_png1 = save_png1 * 255
                save_png0 = save_png0.astype(np.uint8)
                save_png1 = save_png1.astype(np.uint8)
                # Specify the folder names for two targets
                save_dir0 = save_dir + 'target0/'
                save_dir1 = save_dir + 'target1/'
                # Save the segmentation results for two targets in different folders
                save_path0 = save_dir0 + label_name[0]
                save_path1 = save_dir1 + label_name[0]
                if not os.path.exists(save_path0[:save_path0.rfind('/')]):
                    os.makedirs(save_path0[:save_path0.rfind('/')])
                if not os.path.exists(save_path1[:save_path1.rfind('/')]):
                    os.makedirs(save_path1[:save_path1.rfind('/')])
                save_path_s0 = save_dir0 + 's' + label_name[0]
                save_path_s1 = save_dir1 + 's' + label_name[0]
                cv2.imwrite(save_path_s0, save_saliency0)
                cv2.imwrite(save_path_s1, save_saliency1)
                cv2.imwrite(save_path0, save_png0)
                cv2.imwrite(save_path1, save_png1)
            # print(metrics0.precision)

            var_precision = [metrics0.variance('precision'),metrics1.variance('precision')]

            var_recall = [metrics0.variance('recall'),metrics1.variance('recall')]
            var_specificity = [metrics0.variance('specificity'), metrics1.variance('specificity')]
            var_f1 = [metrics0.variance('F1_score'), metrics1.variance('F1_score')]
            var_auc = [metrics0.variance('auc'), metrics1.variance('auc')]
            var_acc = [metrics0.variance('acc'), metrics1.variance('acc')]
            var_iou = [metrics0.variance('iou'), metrics1.variance('iou')]
            var_dice = [metrics0.variance('dice'), metrics1.variance('dice')]
            var_mae = [metrics0.variance('mae'), metrics1.variance('mae')]
            var_hd = [metrics0.variance('hd'), metrics1.variance('hd')]
            print("Test Result:")
            print(args.model_name)
            metrics_result0 = metrics0.mean(len(testloader))
            metrics_result1 = metrics1.mean(len(testloader))
            print("Test Result:") #['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])
            print(var_recall)
            print('recall: %.4f ± %.4f'% (metrics_result0['recall'], var_recall[0])) # delete the space before the %
            print(
                'recall: %.4f ± %.4f, specificity: %.4f± %.4f, precision: %.4f± %.4f, F1_score:%.4f± %.4f, acc: %.4f± %.4f, iou: %.4f± %.4f, mae: %.4f± %.4f, dice: %.4f± %.4f, hd: %.4f± %.4f, auc: %.4f± %.4f'
                % (metrics_result0['recall'], var_recall[0],
                   metrics_result0['specificity'], var_specificity[0], metrics_result0['precision'], var_precision[0],
                   metrics_result0['F1_score'], var_f1[0],
                   metrics_result0['acc'], var_acc[0], metrics_result0['iou'], var_iou[0], metrics_result0['mae'],
                   var_mae[0], metrics_result0['dice'], var_dice[0],
                   metrics_result0['hd'], var_hd[0], metrics_result0['auc'],
                   var_auc[0]))  # add var_auc[0] as the last argument

            print("Test Result for result1:")
            print(
                'recall: %.4f ± %.4f, specificity: %.4f± %.4f, precision: %.4f± %.4f, F1_score:%.4f± %.4f, acc: %.4f± %.4f, iou: %.4f± %.4f, mae: %.4f± %.4f, dice: %.4f± %.4f, hd: %.4f± %.4f, auc: %.4f± %.4f'
                % (metrics_result1['recall'], var_recall[1],
                   metrics_result1['specificity'], var_specificity[1], metrics_result1['precision'], var_precision[1],
                   metrics_result1['F1_score'], var_f1[1],
                   metrics_result1['acc'], var_acc[1], metrics_result1['iou'], var_iou[1], metrics_result1['mae'],
                   var_mae[1], metrics_result1['dice'], var_dice[1],
                   metrics_result1['hd'], var_hd[1], metrics_result1['auc'], var_auc[1]))
            print("total_cost_time:", total_cost_time)
            print("loop_cost_time:", time.time() - all_start)
            evaluation_dir = os.path.sep.join([args.save_dir, 'metrics', args.test_fold + '/' + args.test_dataset + '/'])
            if not os.path.exists(evaluation_dir):
                os.makedirs(evaluation_dir)

            # keys_txt = ''
            metrics_result= metrics0.mean(len(testloader))
            metrics_result['inference_time'] = total_cost_time / len(testloader)
            values_txt = str(args.fold) + '\t'
            for k, v in metrics_result.items():
                if k != 'mae' or k != 'hd':
                    v = 100 * v
                # keys_txt += k + '\t'
                values_txt += '%.2f' % v + '\t'
            text = values_txt + '\n'
            save_path = evaluation_dir + args.model_name +'nst_LARGE+192+64+oct'+ '.txt'
            with open(save_path, 'a+') as f:
                f.write(text)
            print(f'metrics saved in {save_path}')
            print("------------------------------------------------------------------")

        

if __name__ == '__main__':
    args = get_arguments()
    main(args)
