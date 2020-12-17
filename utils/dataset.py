import os
from torch.utils.data import Dataset, DataLoader
import logging
from PIL import Image
import torchvision.transforms as transforms

from utils.utils import test_loader


class PolypDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, imgs_dir, masks_dir, transize=256):
        self.transize = transize
        self.images = [imgs_dir + f for f in os.listdir(imgs_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')]
        self.masks = [masks_dir + f for f in os.listdir(masks_dir) if f.endswith('.png') or f.endswith('.gif')]
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        self.filter_files()
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        mask = self.binary_loader(self.masks[index])

        # transform中进行resize时保留原本比例
        w, h = image.size
        aspect_ratio = h / w
        img_transform = transforms.Compose([
            transforms.Resize(((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16), 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        mask_transform = transforms.Compose([
            transforms.Resize(((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16), 256)),
            transforms.ToTensor()])

        image = img_transform(image)
        mask = mask_transform(mask)
        return {
            'image': image,
            'mask': mask
        }

    @classmethod
    def preprocess(cls, pil_img):
        image = test_loader(pil_img)

        # transform中进行resize时保留原本比例
        w, h = image.size
        aspect_ratio = h / w
        img_transform = transforms.Compose([
            transforms.Resize(((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16), 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        image = img_transform(image)
        return image

    def filter_files(self):
        assert len(self.images) == len(self.masks)
        images = []
        masks = []
        for img_path, mask_path in zip(self.images, self.masks):
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            if img.size == mask.size:
                images.append(img_path)
                masks.append(mask_path)
        self.images = images
        self.masks = masks

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if h < self.transize or w < self.transize:
            h = max(h, self.transize)
            w = max(w, self.transize)
            return img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        else:
            return img, mask

    def __len__(self):
        return self.size


class test_dataset:
    def __init__(self, imgs_dir, masks_dir, transize=256):
        self.transize = transize
        self.images = [imgs_dir + f for f in os.listdir(imgs_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [masks_dir + f for f in os.listdir(masks_dir) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        mask = self.binary_loader(self.gts[self.index])
        # transform中进行resize时保留原本比例
        w, h = image.size
        aspect_ratio = h / w
        img_transform = transforms.Compose([
            transforms.Resize(((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16), 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        mask_transform = transforms.Compose([
            transforms.Resize(((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16), 256)),
            transforms.ToTensor()])
        image = img_transform(image)
        mask = mask_transform(mask)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, mask, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')