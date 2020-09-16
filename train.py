import argparse
import logging
import os
import sys
# import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
import torch.nn.functional as F
from tqdm import tqdm

from eval import eval_net
from networks import UNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, ResUnetPlusPlus, PraNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from utils.utils import Structure_loss
from torch.utils.data import DataLoader, random_split

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

dir_img = './data/CVCpolyp/augment/imgs/'
dir_mask = './data/CVCpolyp/augment/masks/'
dir_checkpoint = 'checkpoints_CVCpolyp/'

# dir_img = './data/cars/imgs/'
# dir_mask = './data/cars/masks/'
# dir_checkpoint = 'checkpoints_cars/'


# dir_img = './data/blood_vessel/imgs/'
# dir_mask = './data/blood_vessel/masks/'
# dir_checkpoint = 'checkpoints_blood_vessel/'

def train_net(net,
              network_name,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):
    dataset = BasicDataset(dir_img, dir_mask, img_scale, mask_suffix="")
    n_val = int(len(dataset) * val_percent)  # 验证集图像个数
    n_train = len(dataset) - n_val  # 训练集图像个数
    train, val = random_split(dataset, [n_train, n_val])  # 根据大小。划分训练集与验证集
    # 加载训练集与验证集，获取一个批次的数据
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    # tensorboard
    writer = SummaryWriter(comment=f'_BS={batch_size}_Epoch={epochs}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    # 选择梯度下降的优化器
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(list(net.parameters()), lr=lr, betas=(0.5, 0.999))
    # 训练过程中自动调整学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    # 选择损失函数
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        # criterion = nn.BCEWithLogitsLoss()
        criterion = Structure_loss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        size_rates = [0.75, 1, 1.25]
        with tqdm(total=n_train * 3, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                for rate in size_rates:
                    imgs = batch['image']
                    true_masks = batch['mask']
                    assert imgs.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    # rescale
                    new_height = int(round(192*rate/32)*32)
                    new_width = int(round(256*rate/32)*32)
                    if rate != 1:
                        imgs = F.upsample(imgs, size=(new_height, new_width), mode='bilinear', align_corners=True)
                        true_masks = F.upsample(true_masks, size=(new_height, new_width), mode='bilinear', align_corners=True)

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if net.n_classes == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)
                    # 获得输出并计算损失,PraNet的损失计算方式不同，这里有所区分
                    if network_name == 'PraNet':
                        masks_pred_4, masks_pred_3, masks_pred_2, masks_pred = net(imgs)
                        loss5 = criterion(masks_pred_4, true_masks)
                        loss4 = criterion(masks_pred_3, true_masks)
                        loss3 = criterion(masks_pred_2, true_masks)
                        loss2 = criterion(masks_pred, true_masks)
                        loss = loss2 + loss3 + loss4 + loss5
                    else:
                        masks_pred = net(imgs)
                        loss = criterion(masks_pred, true_masks)
                    epoch_loss += loss.item()
                    writer.add_scalar('Loss/train', loss.item(), global_step)

                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    # 执行梯度下降更新权重
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    '''for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    '''
                    # 验证集测试的dice分数
                    # train_score = eval_net(net, train_loader, device, network_name)
                    val_score = eval_net(net, val_loader, device, network_name)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        print(" ")
                        # logging.info('Training Dice Coeff: {}'.format(train_score))
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        print(" ")
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


# argparse 模块可以让人轻松编写用户友好的命令行接口
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--network', metavar='N', type=str, default="NestedUNet",
                        help='choice of network: UNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, '
                             'ResUnetPlusPlus, PraNet', dest='network')
    parser.add_argument('-e', '--epochs', metavar='E', type=str, default=8,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


# 正式开始训练
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    # 选择想要训练的网络
    if args.network == 'U_net':
        # net = UNet(n_channels=3, n_classes=1, bilinear=False)
        net = U_Net(n_channels=3, n_classes=1, bilinear=False)
    if args.network == 'R2U_Net':
        net = R2U_Net(n_channels=3, n_classes=1, bilinear=False)
    if args.network == 'AttU_Net':
        net = AttU_Net(n_channels=3, n_classes=1, bilinear=False)
    if args.network == 'R2AttU_Net':
        net = R2AttU_Net(n_channels=3, n_classes=1, bilinear=False)
    if args.network == 'NestedUNet':
        net = NestedUNet(n_channels=3, n_classes=1, bilinear=False)
    if args.network == 'ResUnetPlusPlus':
        net = ResUnetPlusPlus(n_channels=3, n_classes=1, bilinear=False)
    if args.network == 'PraNet':
        net = PraNet(n_channels=3, n_classes=1, bilinear=False)

    logging.info(f'Network:\t{args.network}\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  network_name=args.network,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
