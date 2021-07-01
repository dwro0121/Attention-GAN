import os

import torch
import argparse
from torch.backends import cudnn

from models.gan import D, G
from utils.tool import loadmodel
from torch.autograd import Variable
from torchvision.utils import save_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--z_dim", type=int, default=64, help="dim of latent vector z (default: 128)"
    )
    parser.add_argument(
        "--num_gen",
        type=int,
        default=0,
        help="number of generate images. if insert 0, generate images on grid (default: 0) ",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="SA",
        help="version of attention [SA, CCA, YLA] (default: SA) ",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CelebA",
        help="dataset for train, need to place under ./data/ (default: CelebA)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./saved_models/",
        help="path for load models (default: ./saved_models/)",
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default="./img_samples/",
        help="path for save sample images (default: ./img_samples/)",
    )
    parser.add_argument(
        "--gen_path",
        type=str,
        default="./img_gen/",
        help="path for save generated images (default: ./img_gen/)",
    )
    parser.add_argument(
        "--load",
        type=int,
        default=0,
        help="load saved model (default: 0)",
    )
    cfg = parser.parse_args()
    cudnn.benchmark = True

    cfg.model_path += cfg.version + "_" + cfg.dataset
    cfg.gen_path += cfg.version + "_" + cfg.dataset
    cfg.sample_path += cfg.version + "_" + cfg.dataset

    if not os.path.exists(cfg.gen_path):
        os.makedirs(cfg.gen_path)
    if not os.path.exists(cfg.sample_path):
        os.makedirs(cfg.sample_path)

    G = G(cfg.z_dim, cfg.version).cuda()

    if cfg.load != 0:
        G_path = cfg.model_path + "/" + str(cfg.load) + "_G.pth"
        G_state_dict = torch.load(G_path)
        G.load_state_dict(G_state_dict)
        G = G.cuda()
    if cfg.num_gen == 0:
        for i in range(10):
            z = Variable(torch.randn(64, cfg.z_dim).cuda())
            if i == 0:
                generated_imgs = G(z).cpu().data
            else:
                generated_imgs = torch.cat([generated_imgs, G(z).cpu().data], 0)
        save_image(
            ((generated_imgs + 1) / 2).clamp_(0, 1),
            os.path.join(
                cfg.sample_path,
                "generated_grid.png",
            ),
            10,
            5,
        )
    else:
        for i in range(cfg.num_gen):
            z = Variable(torch.randn(64, cfg.z_dim).cuda())
            generated_imgs = G(z).cpu().data
            for j in range(64):
                save_image(
                    ((generated_imgs[j] + 1) / 2).clamp_(0, 1),
                    os.path.join(
                        cfg.gen_path,
                        str(64 * i + j) + ".png",
                    ),
                )
    print("generate completed!")
