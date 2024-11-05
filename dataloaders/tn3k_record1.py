import torch.utils.data as data
import PIL.Image as Image
import os
import json
import numpy as np
import torch
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F


def make_dataset(root, seed, name):
    imgs = []
    img_labels = {}
    img_names = os.listdir(root + name+'-image/')
    img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))

    for i in seed:
        img_name = img_names[i]
        img = os.path.join(root +'/'+ name+ '-image/', img_name)
        mask = os.path.join(root +'/'+ name+ '-mask/', img_name)
        imgs.append((img, mask))
    return imgs


def make_testset(root):
    imgs = []
    img_labels = {}
    img_names = os.listdir(root +'/test-image/')
    img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))

    for img_name in img_names:
        img = os.path.join(root +'/test-image/', img_name)
        mask = os.path.join(root +'/test-mask/', img_name)
        imgs.append((img, mask))
    return imgs


class TN3K(data.Dataset):
    def __init__(self, mode, transform=None, return_size=False, fold=0):
        self.mode = mode
        # FIXME:for test, 记得改回原来的 ./data/tn3k
        root = './data/tn3k/'
        trainval = json.load(open(root + 'tn3k-trainval-fold'+str(fold)+'.json', 'r'))
        if mode == 'train':
            imgs = make_dataset(root, trainval['train'], 'trainval')
        elif mode == 'val':
            imgs = make_dataset(root, trainval['val'], 'trainval')
        elif mode == 'test':
            imgs = make_testset(root)

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, item):
        if True:
            image_path, label_path = self.imgs[item]
            assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
            assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

            image = Image.open(image_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            # image.show()
            # label.show()
            # label=np.array(label)
            # label = label / label.max()
            #
            # label = Image.fromarray(label.astype(np.uint8))
            w, h = image.size
            size = (h, w)

            sample = {'image': image, 'label': label}

            if self.transform:

                p_transform = random.random()
                aspect_ratio = image.size[1] / image.size[0]
                Transform = []
                Transform_GT = []

                if (self.mode == 'train') and p_transform <= 0.9:
                    # resize操作想不明白有啥用
                    # Transform.append(
                    #     T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.BICUBIC))  # 双三次
                    # Transform_GT.append(
                    #     T.Resize((int(ResizeRange * aspect_ratio), ResizeRange), interpolation=Image.NEAREST))  # 最近邻

                    RotationDegree = random.randint(0, 7)
                    rotation = [0,90,180,270,45,135,215,305]
                    RotationDegree = rotation[RotationDegree]

                    if (RotationDegree == 90) or (RotationDegree == 270):
                        aspect_ratio = 1 / aspect_ratio

                    Transform.append(T.RandomRotation((RotationDegree, RotationDegree)))
                    Transform_GT.append(T.RandomRotation((RotationDegree, RotationDegree)))

                    # 在大旋转间隔基础上,微小调整旋转角度
                    RotationRange = random.randint(-10, 10)
                    Transform.append(T.RandomRotation((RotationRange, RotationRange)))
                    Transform_GT.append(T.RandomRotation((RotationRange, RotationRange)))

                    CropRange = random.randint(512, 640)
                    Transform.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
                    # Transform.append(T.CenterCrop((int(CropRange ), CropRange)))
                    Transform_GT.append(T.CenterCrop((int(CropRange * aspect_ratio), CropRange)))
                    #
                    Transform = T.Compose(Transform)
                    Transform_GT = T.Compose(Transform_GT)

                    if random.random()>0.5 :

                        sample['image'] = Transform(sample['image'])
                        sample['label'] = Transform(sample['label'])
                        # sample['image'].show()
                        # sample['label'].show()



                    # crop
                    ShiftRange_left = random.randint(0, 20)
                    ShiftRange_upper = random.randint(0, 20)
                    ShiftRange_right = image.size[0] - random.randint(0, 20)
                    ShiftRange_lower = image.size[1] - random.randint(0, 20)

                    if random.random() > 0.5:

                        sample['image'] = sample['image'].crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))
                        sample['label'] = sample['label'].crop(box=(ShiftRange_left, ShiftRange_upper, ShiftRange_right, ShiftRange_lower))

                    if random.random() < 0.5:
                        sample['image'] = F.vflip(sample['image'])
                        sample['label'] = F.vflip(sample['label'])


                # sample['image'].show()
                # sample['label'].show()
                    # Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)
                    #
                    # image = Transform(image)
                # print(sample['image'].size(),sample['label'].size())

                sample['label'] = np.array(sample['label'])
                sample['label'] = sample['label'] / sample['label'].max()

                sample['label'] = Image.fromarray(sample['label'].astype(np.uint8))

                sample = self.transform(sample)
                    # Transform = []
                    # Transform_GT = []

            if self.return_size:
                sample['size'] = torch.tensor(size)

            label_name = os.path.basename(label_path)
            sample['label_name'] = label_name

            return sample
        # else:
        #     image_path = self.imgs[item]
        #     image = Image.open(image_path).convert('RGB')
        #     label = Image.open(image_path)
        #     sample = {'image': image, 'label': label}
        #     w, h = image.size
        #     size = (h, w)
        #
        #     if self.transform:
        #         sample = self.transform(sample)
        #     if self.return_size:
        #         sample['size'] = torch.tensor(size)
        #
        #     label_name = os.path.basename(image_path)
        #     sample['label_name'] = label_name
        #
        #     return sample

    def __len__(self):
        return len(self.imgs)
