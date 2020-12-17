import os
import cv2
import PIL
from imgaug import augmenters as iaa

original_imgs_dir = "../data/CVCpolyp/imgs/"
original_masks_dir = "../data/CVCpolyp/masks/"
aug_imgs_dir = "../data/CVCpolyp/augment/imgs/"
aug_masks_dir = "../data/CVCpolyp/augment/masks/"

original_imgs = os.listdir(original_imgs_dir)
original_masks = os.listdir(original_masks_dir)
# print(len(original_imgs), original_imgs)
# print(len(original_masks), original_masks)

# 对于img和mask分别定义增强过程
seq1 = iaa.Sequential([
    iaa.Fliplr(),  # 水平翻转图像
    iaa.Flipud(),
])
seq2 = iaa.Sequential([
    iaa.Fliplr(),  # 水平翻转图像
    iaa.Flipud(),
])

img_list = []
mask_list = []
for i in range(len(original_imgs)):
    img = cv2.imread(original_imgs_dir + original_imgs[i])
    img_list.append(img)
    # 对于mask,要以灰度图读取，不然保存时会变成3通道
    mask = cv2.imread(original_masks_dir + original_masks[i], cv2.IMREAD_GRAYSCALE)
    mask_list.append(mask)
print("read over...")
imgaug_list = seq1.augment_images(img_list)
maskaug_list = seq2.augment_images(mask_list)

# 将增强后的图片写入到新的文件夹
for i in range(len(original_imgs)):
    cv2.imwrite(aug_imgs_dir + original_imgs[i], imgaug_list[i])
    cv2.imwrite(aug_masks_dir + original_masks[i], maskaug_list[i])

