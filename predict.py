import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from networks import UNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, ResUnetPlusPlus, PraNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                network_name,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        if network_name == 'PraNet':
            output4, output3, output2, output = net(img)
        else:
            output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./param_initial/PraNet-29.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-n', '--network', metavar='N', type=str, default="PraNet",
                        help='choice of network: UNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, '
                             'ResUnetPlusPlus, PraNet', dest='network')

    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', default='./data/CVCpolyp/pred/604.png',
                        help='filenames of input images')
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', default='./data/CVCpolyp/pred/604_PraNet.png',
                        help='Filenames of ouput images')

    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=True)
    parser.add_argument('--no-save', '-ns', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = [args.input]
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len([args.output]):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = [args.output]

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = [args.input]
    out_files = get_output_filenames(args)

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

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        mask = predict_img(net=net,
                           network_name=args.network,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)