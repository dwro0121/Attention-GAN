import os
import argparse
from torch.backends import cudnn
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torchvision.utils import save_image
import time
from collections import OrderedDict


def cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--z_dim", type=int, default=64, help="dim of latent vector z (default: 128)"
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
        default="celebA",
        help="dataset for train, need to place under ./data/ (default: celebA)",
    )
    parser.add_argument(
        "--total_epoch",
        type=int,
        default=100,
        help="how many epochs to train (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="global batch size for train (default: 64)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of cpu workers (default: 2)",
    )
    parser.add_argument(
        "--parallel",
        type=str2bool,
        default=False,
        help="parallel[Multi-GPU] train or not (default: False)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/",
        help="path for data files (default: ./data/)",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./logs/",
        help="path for logs (default: ./logs/)",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./saved_models/",
        help="path for save models (default: ./saved_models/)",
    )
    parser.add_argument(
        "--sample_path",
        type=str,
        default="./img_samples/",
        help="path for save sample images (default: ./img_samples/)",
    )
    parser.add_argument(
        "--model_save_step",
        type=int,
        default=1,
        help="need how many epochs to save model (default: 1)",
    )

    parser.add_argument(
        "--load",
        type=int,
        default=0,
        help="load saved model (default: 0)",
    )

    cfg = parser.parse_args()
    cudnn.benchmark = True

    cfg.image_path += cfg.dataset
    cfg.log_path += cfg.version + "_" + cfg.dataset
    cfg.model_save_path += cfg.version + "_" + cfg.dataset
    cfg.sample_path += cfg.version + "_" + cfg.dataset

    if not os.path.exists(cfg.model_save_path):
        os.makedirs(cfg.model_save_path)
    if not os.path.exists(cfg.sample_path):
        os.makedirs(cfg.sample_path)
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    return cfg


def train_one_epoch(cfg, G, D, optim_g, optim_d, fixed_z, c_loss, data_loader, epoch):

    with tqdm(
        total=len(data_loader),
        desc=f"Epoch {epoch+1}/{cfg.total_epoch}",
        postfix=dict,
        mininterval=0.5,
    ) as pbar:
        for i, real_imgs in enumerate(data_loader):
            step_time = time.time()
            D.train()
            G.train()

            real_imgs = Variable(real_imgs.cuda())
            z = Variable(torch.randn(cfg.batch_size, cfg.z_dim).cuda())
            real_d = D(real_imgs)
            fake_imgs = G(z)
            fake_d = D(fake_imgs)
            loss_real_d = torch.nn.ReLU()(1.0 - real_d).mean()
            loss_fake_d = torch.nn.ReLU()(1.0 + fake_d).mean()
            loss_d = loss_real_d + loss_fake_d

            optim_d.zero_grad()
            optim_g.zero_grad()
            loss_d.backward()
            optim_d.step()

            z = Variable(torch.randn(cfg.batch_size, cfg.z_dim).cuda())
            fake_imgs = G(z)
            fake_d = D(fake_imgs)
            loss_g = -fake_d.mean()

            optim_d.zero_grad()
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

            waste_time = time.time() - step_time

            pbar.set_postfix(
                **{
                    "Real D Loss": loss_real_d.item(),
                    "Fake D Loss": loss_fake_d.item(),
                    "D Loss": loss_d.item(),
                    "G Loss": loss_g.item(),
                    "s/step": waste_time,
                }
            )

            pbar.update(1)
    if epoch % cfg.model_save_step == 0:
        fake_imgs = G(fixed_z)
        save_image(
            ((fake_imgs.data + 1) / 2).clamp_(0, 1),
            os.path.join(
                cfg.sample_path,
                "sample_{}.png".format(epoch + 1),
            ),
            10,
            5,
        )
        torch.save(
            G.state_dict(),
            os.path.join(
                cfg.model_save_path,
                "{}_G.pth".format(epoch + 1),
            ),
        )
        torch.save(
            D.state_dict(),
            os.path.join(
                cfg.model_save_path,
                "{}_D.pth".format(epoch + 1),
            ),
        )


def str2bool(line):
    return line.lower() in ("true")


def loadmodel(model, path):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model.cuda()
