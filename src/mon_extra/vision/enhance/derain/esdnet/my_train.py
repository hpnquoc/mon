#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import random
import time

import torch.optim
import torch.optim as optim
from spikingjelly.activation_based import functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mon
import utils
from dataset_load import Dataload
from losses import *
from model import model

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark     = True

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Train

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args: argparse.Namespace):
    # General config
    data_dir         = mon.ROOT_DIR / args.data_dir
    save_dir         = mon.Path(args.save_dir)
    weights          = args.weights
    device           = mon.set_device(args.device)
    epochs           = args.epochs
    verbose          = args.verbose
    mode             = args.mode
    patch_size_train = args.patch_size_train
    patch_size_test  = args.patch_size_test
    batch_size       = args.batch_size
    start_lr         = args.lr
    end_lr           = args.min_lr
    warmup_epochs    = args.warmup_epochs
    clip_grad        = args.clip_grad
    use_amp          = args.use_amp
    num_workers      = args.num_workers
    
    # Directory
    weights_dir = save_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # print number of model
    # get_parameter_number(model_restoration)
    # device_ids = 0
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
        
    # Model
    model_restoration = model
    model_restoration.cuda()
    functional.set_step_mode(model_restoration, step_mode="m")
    functional.set_backend(model_restoration,   backend="cupy")
    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)
        
    # Loss
    # criterion = nn.MSELoss().cuda()
    criterion_ssim = utils.SSIM().cuda()
    # criterion_L1 = nn.SmoothL1Loss().cuda()
    criterion_psnr = PSNRLoss().cuda()
    
    # Optimizer
    optimizer        = optim.AdamW(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - warmup_epochs, eta_min=end_lr)
    scheduler        = mon.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    
    # Data I/O
    train_dataset = Dataload(data_dir=str(data_dir / "train"), patch_size=patch_size_train)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        drop_last   = False,
        pin_memory  = True
    )
    
    val_dataset = Dataload(data_dir=str(data_dir / "val"), patch_size=patch_size_test)
    val_loader  = torch.utils.data.DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 1,
        drop_last   = False,
        pin_memory  = True
    )
    
    # Training
    best_psnr  = 0
    best_epoch = 0
    writer     = SummaryWriter(weights_dir)
    iter       = 0
    scaler     = torch.cuda.amp.GradScaler()
    
    with mon.get_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(epochs),
            total       = epochs,
            description = f"[bright_yellow] Training"
        ):
            epoch_start_time   = time.time()
            epoch_loss         = 0
            train_psnr_val_rgb = []
            scaled_loss        = 0
            model_restoration.train()
            # scheduler.step()
            
            # Train
            for i, data in enumerate(tqdm(train_loader, unit="img"), 0):
                for param in model_restoration.parameters():
                    param.grad = None
                input_   = data[0].cuda()
                target_  = data[1].cuda()
                restored = model_restoration(input_)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        ssim = criterion_ssim(restored, target_)
                        loss = 1 - ssim
                    scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    functional.reset_net(model_restoration)
                else:
                    # L1_Loss = criterion_L1(restored, target_)
                    ssim = criterion_ssim(restored, target_)
                    psnr = criterion_psnr(restored, target_)
                    loss = 1 - ssim
                    loss.backward()
                    scaled_loss += loss.item()
                    # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                    optimizer.step()
                    functional.reset_net(model_restoration)
                torch.cuda.synchronize()
                epoch_loss += loss.item()
                iter       += 1
                for res, tar in zip(restored, target_):
                    train_psnr_val_rgb.append(utils.torchPSNR(res, tar))
                psnr_train = torch.stack(train_psnr_val_rgb).mean().item()
                
                writer.add_scalar("loss/iter_loss",  loss.item(), iter)
                writer.add_scalar("loss/epoch_loss", epoch_loss, epoch)
                writer.add_scalar("lr/epoch_loss",   scheduler.get_lr()[0], epoch)
                
            # Evaluation
            if epoch % 1 == 0:
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate(tqdm(val_loader, unit="img"), 0):
                    input_ = data_val[0].cuda()
                    target = data_val[1].cuda()
                    
                    with torch.no_grad():
                        restored = model_restoration(input_)
                    functional.reset_net(model_restoration)
        
                    for res, tar in zip(restored, target):
                        psnr_val_rgb.append(utils.torchPSNR(res, tar))
                
                psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
                writer.add_scalar("val/psnr", psnr_val_rgb, epoch)
                if psnr_val_rgb > best_psnr:
                    best_psnr  = psnr_val_rgb
                    best_epoch = epoch
                    torch.save(model_restoration.state_dict(), str(weights_dir / "esdnet_best.pt"))
                print("[epoch %d Training PSNR: %.4f --- best_epoch %d Test_PSNR %.4f]" % (epoch, psnr_train, best_epoch, best_psnr))
            
            # Save model
            if epoch % 50 == 0:
                torch.save(
                    {
                        "epoch"     : epoch,
                        "state_dict": model_restoration.state_dict(),
                        "optimizer" : optimizer.state_dict()
                    },
                    str(weights_dir / f"esdnet_epoch_{epoch}.pt")
                )
            torch.save(
                {
                    "epoch"     : epoch,
                    "state_dict": model_restoration.state_dict(),
                    "optimizer" : optimizer.state_dict()
                },
                str(weights_dir / "esdnet_last.pt")
            )
            scheduler.step()
            print("-" * 150)
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tTrain_PSNR: {:.4f}\tSSIM: {:.4f}\tLearningRate {:.8f}\tTest_PSNR: {:.4f}".format(
                    epoch, time.time() - epoch_start_time, loss.item(), psnr_train, ssim, scheduler.get_lr()[0],
                    best_psnr, ))
            print("-" * 150)
        writer.close()
        
# endregion


# region Main

def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()

# endregion
