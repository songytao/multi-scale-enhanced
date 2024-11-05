import torch.utils.data as data
import PIL.Image as Image
import os
import json
import numpy as np
import torch
import random
from torchvision import transforms as T

import torchvision.transforms as standard_transforms

from torchvision.transforms import functional as F


def make_dataset(root, seed, name, num_classes):
    imgs = []
    img_labels = {}
    img_names = os.listdir(root + name + '-image/')
    img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))

    for i in seed:
        # print(i,len(img_names))
        img_name = img_names[i-1]
        img = os.path.join(root + '/' + name + '-image/', img_name)
        if num_classes ==1:
            mask = os.path.join(root + '/' + name + '-mask/', img_name)
            imgs.append((img, mask))
        else:
            mask0 = os.path.join(root + '/' + name + '-mask0/', img_name)
            mask1 = os.path.join(root + '/' + name + '-mask1/', img_name)
            imgs.append((img, mask0, mask1))
    return imgs


def make_testset(root,num_classes):
    imgs = []
    img_labels = {}
    img_names = os.listdir(root + '/test-image/')
    img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))

    for img_name in img_names:
        img = os.path.join(root + '/test-image/', img_name)
        if num_classes ==1:
            mask = os.path.join(root + '/test-mask/', img_name)
            imgs.append((img, mask))
        else:
            mask0 = os.path.join(root + '/test-mask0/', img_name)
            mask1 = os.path.join(root + '/test-mask1/', img_name)
            imgs.append((img, mask0, mask1))
    return imgs


class TN3K(data.Dataset):
    def __init__(self, mode, transform=None, return_size=False, fold=0 , dataname = None, num_classes = 1):
        self.mode = mode
        # FIXME:for test, 记得改回原来的 ./data/tn3k
        if dataname == None:
            root = '/home/user3/new_code/new_code/data' + 'tn3k'+'/'
        else:
            root = '/home/user3/new_code/new_code/data/'+dataname+'/'

        if mode == 'train':
            trainval = json.load(open(root + 'tn3k-trainval-fold' + str(fold) + '.json', 'r'))
            imgs = make_dataset(root, trainval['train'], 'trainval', num_classes)
        elif mode == 'val':
            trainval = json.load(open(root + 'tn3k-trainval-fold' + str(fold) + '.json', 'r'))
            imgs = make_dataset(root, trainval['val'], 'trainval', num_classes)
        elif mode == 'test':
            imgs = make_testset(root,num_classes)

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size
        self.num_classes = num_classes

    def __getitem__(self, item):
        if self.num_classes == 1:
            image_path, label_path = self.imgs[item]
            assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
            assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

            image = Image.open(image_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            # image.show()
            # label.show()
            label = np.array(label)
            label = label / label.max()

            label = Image.fromarray(label.astype(np.uint8))
            w, h = image.size
            size = (h, w)

            sample = {'image': image, 'label': label}

            if self.transform:

                p_transform = random.random()
                aspect_ratio = image.size[1] / image.size[0]
                Transform = []
                Transform_GT = []

                if (self.mode == 'train') and p_transform <= 0.65:

                    # Transform.append(
                    #     T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.BICUBIC))  # 双三次
                    # Transform_GT.append(
                    #     T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.NEAREST))  # 最近邻

                    RotationDegree = random.randint(0, 7)
                    rotation = [0, 90, 180, 270, 45, 135, 215, 305]
                    RotationDegree = rotation[RotationDegree]

                    if (RotationDegree == 90) or (RotationDegree == 270):
                        aspect_ratio = 1 / aspect_ratio

                    Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))
                    Transform_GT.append(T.RandomRotation((RotationDegree, RotationDegree)))

                    # 在大旋转间隔基础上,微小调整旋转角度
                    RotationRange = random.randint(-10, 10)
                    Transform.append(T.RandomRotation((RotationRange, RotationRange)))
                    Transform_GT.append(T.RandomRotation((RotationRange, RotationRange)))

                    CropRange = random.randint(190, 217)
                    Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
                    # Transform.append(T.CenterCrop((int(CropRange ), CropRange)))
                    Transform_GT.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
                    #
                    Transform = T.Compose(Transform)
                    Transform_GT = T.Compose(Transform_GT)

                    # crop
                    ShiftRange_left = random.randint(0, 20)
                    ShiftRange_upper = random.randint(0, 20)
                    ShiftRange_right = image.size[0] - random.randint(0, 20)
                    ShiftRange_lower = image.size[1] - random.randint(0, 20)

                    rate = random.random()

                    if rate < 0.33:

                        sample['image'] = Transform(sample['image'])
                        sample['label'] = Transform(sample['label'])
                        # sample['image'].show()
                        # sample['label'].show()





                    elif rate < 0.66:

                        sample['image'] = sample['image'].crop(
                            box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
                        sample['label'] = sample['label'].crop(
                            box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

                    else:
                        sample['image'] = F.vflip(sample['image'])
                        sample['label'] = F.vflip(sample['label'])


                sample = self.transform(sample)
                # Transform = []
                # Transform_GT = []

            if self.return_size:
                sample['size'] = torch.tensor(size)

            label_name = os.path.basename(label_path)
            sample['label_name'] = label_name

            return sample
        else:

            image_path, label_path0, label_path1 = self.imgs[item]
            assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
            assert os.path.exists(label_path0), ('{} does not exist'.format(label_path0))

            image = Image.open(image_path).convert('RGB')
            label0 = Image.open(label_path0).convert('L')
            label1 = Image.open(label_path1).convert('L')

            label0 = np.array(label0)
            label0= label0 / label0.max()

            label1 = np.array(label1)
            label1 = label1 / label1.max()

            label0 = Image.fromarray(label0.astype(np.uint8))
            label1 = Image.fromarray(label1.astype(np.uint8))
            w, h = image.size
            size = (h, w)
            # print(type(label0))
            #
            # transform1 = standard_transforms.ToTensor()
            # label0 = transform1(label0)
            # label1 = transform1(label1)
            # label = torch.stack([label0, label1], dim=0)
            # transform2 = standard_transforms.ToPILImage()
            # print(label.shape)
            # label = transform2(label)

            sample = {'image': image, 'label0': label0, 'label1': label1 }

            if self.transform:

                p_transform = random.random()
                aspect_ratio = image.size[1] / image.size[0]
                Transform = []
                Transform_GT = []

                if (self.mode == 'train') and p_transform <= 0.7:


                    RotationDegree = random.randint(0, 7)
                    rotation = [0, 90, 180, 270, 45, 135, 215, 305]
                    RotationDegree = rotation[RotationDegree]

                    if (RotationDegree == 90) or (RotationDegree == 270):
                        aspect_ratio = 1 / aspect_ratio

                    Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))
                    Transform_GT.append(T.RandomRotation((RotationDegree, RotationDegree)))

                    # 在大旋转间隔基础上,微小调整旋转角度
                    RotationRange = random.randint(-10, 10)
                    Transform.append(T.RandomRotation((RotationRange, RotationRange)))
                    Transform_GT.append(T.RandomRotation((RotationRange, RotationRange)))

                    CropRange = random.randint(190, 217)
                    Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
                    # Transform.append(T.CenterCrop((int(CropRange ), CropRange)))
                    Transform_GT.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
                    #
                    Transform = T.Compose(Transform)
                    Transform_GT = T.Compose(Transform_GT)

                    # crop
                    ShiftRange_left = random.randint(0, 20)
                    ShiftRange_upper = random.randint(0, 20)
                    ShiftRange_right = image.size[0] - random.randint(0, 20)
                    ShiftRange_lower = image.size[1] - random.randint(0, 20)

                    rate = random.random()

                    if rate < 0.33:

                        sample['image'] = Transform(sample['image'])
                        sample['label0'] = Transform(sample['label0'])
                        sample['label1'] = Transform(sample['label1'])

                        # sample['image'].show()
                        # sample['label'].show()





                    elif rate < 0.66:

                        sample['image'] = sample['image'].crop(
                            box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
                        sample['label0'] = sample['label0'].crop(
                            box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

                        sample['label1'] = sample['label1'].crop(
                            box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))


                    else:
                        sample['image'] = F.vflip(sample['image'])
                        sample['label0'] = F.vflip(sample['label0'])
                        sample['label1'] = F.vflip(sample['label1'])



                sample = self.transform(sample)


            if self.return_size:
                sample['size'] = torch.tensor(size)
            sample['label'] = torch.stack([sample['label0'], sample['label1']], dim=0)
            sample['label'] = torch.squeeze(sample['label'])
            label_name = os.path.basename(label_path0)
            sample['label_name'] = label_name

            return sample


    def __len__(self):
        return len(self.imgs)
