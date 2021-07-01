import os

import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import transforms

from utils.dataset import CustomDataset
from models.gan import D, G
from utils.tool import *

if __name__ == "__main__":
    cfg = cfg()
    G = G(cfg.z_dim, cfg.version).cuda()
    D = D(cfg.version).cuda()

    if cfg.parallel:
        if cfg.load != 0:
            G_path = cfg.model_save_path + "/" + str(cfg.load) + "_G.pth"
            D_path = cfg.model_save_path + "/" + str(cfg.load) + "_D.pth"
            G = loadmodel(G, G_path).cuda()
            D = loadmodel(D, D_path).cuda()
        G = nn.DataParallel(G, [0, 1, 2, 3, 4, 5, 6])
        D = nn.DataParallel(D, [0, 1, 2, 3, 4, 5, 6])
    else:
        if cfg.load != 0:
            G_path = cfg.model_save_path + "/" + str(cfg.load) + "_G.pth"
            D_path = cfg.model_save_path + "/" + str(cfg.load) + "_D.pth"
            G_state_dict = torch.load(G_path)
            D_state_dict = torch.load(D_path)
            G.load_state_dict(G_state_dict)
            D.load_state_dict(D_state_dict)
            G = G.cuda()
            D = D.cuda()

    optim_g = torch.optim.Adam(
        filter(lambda p: p.requires_grad, G.parameters()),
        1e-4,
        [0, 0.9],
    )
    optim_d = torch.optim.Adam(
        filter(lambda p: p.requires_grad, D.parameters()),
        4e-4,
        [0, 0.9],
    )
    c_loss = torch.nn.CrossEntropyLoss()
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CustomDataset(cfg.image_path, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    fixed_z = Variable(torch.randn(10, cfg.z_dim).cuda())
    for epoch in range(cfg.load, cfg.total_epoch):
        train_one_epoch(
            cfg, G, D, optim_g, optim_d, fixed_z, c_loss, data_loader, epoch
        )
    print("train completed!")
