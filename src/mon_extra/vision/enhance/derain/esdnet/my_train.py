#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import random
import time

import torch.optim
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import mon
import utils
from dataset_load import Dataload
from losses import *
from model import model
from spikingjelly.activation_based import functional

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark     = True

# A workaround for the bug in numpy >= 1.2.4
np.int   = np.int32
np.float = np.float64
np.bool  = np.bool_

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Train

def train(args: argparse.Namespace):
    # General config
    data             = args.data
    data_dir         = mon.ROOT_DIR / args.data_dir
    fullname         = args.fullname
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
    
    # Model
    model_restoration = model
    model_restoration.to(device)
    if weights is not None and mon.Path(weights).is_weights_file():
        model_restoration.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    functional.set_step_mode(model_restoration, step_mode="m")
    functional.set_backend(model_restoration,   backend="cupy")
    
    # Loss
    # criterion = nn.MSELoss().to(device)
    criterion_ssim = utils.SSIM().to(device)
    # criterion_L1 = nn.SmoothL1Loss().to(device)
    criterion_psnr = PSNRLoss().to(device)
    
    # Optimizer
    optimizer        = optim.AdamW(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - warmup_epochs, eta_min=end_lr)
    scheduler        = mon.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    
    # Data I/O
    train_dir     = data_dir / "train"
    train_dataset = Dataload(data_dir=train_dir, patch_size=patch_size_train)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        drop_last   = False,
        pin_memory  = True
    )
    
    if (data_dir / "val").exists():
        val_dir = data_dir / "val"
    else:
        val_dir = data_dir / "test"
    val_dataset = Dataload(data_dir=val_dir, patch_size=patch_size_test)
    val_loader  = torch.utils.data.DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 1,
        drop_last   = False,
        pin_memory  = True
    )
    
    # Training
    writer          = SummaryWriter(weights_dir)
    scaler          = torch.cuda.amp.GradScaler()
    best_psnr       = 0
    best_ssim       = 0
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    iter            = 0
    
    for epoch in range(0, epochs):
        epoch_start_time   = time.time()
        epoch_loss         = 0
        scaled_loss        = 0
        train_psnr_val_rgb = []
        model_restoration.train()
        # scheduler.step()
        
        # Train
        with mon.get_progress_bar() as pbar:
            for i, data in pbar.track(
                sequence    = enumerate(train_loader),
                total       = len(train_loader),
                description = f"[bright_yellow] Training"
            ):
                for param in model_restoration.parameters():
                    param.grad = None
                image    = data[0].to(device)
                ref      = data[1].to(device)
                restored = model_restoration(image)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        ssim = criterion_ssim(restored, ref)
                        # psnr = criterion_psnr(restored, ref)
                        loss = 1 - ssim
                    scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    functional.reset_net(model_restoration)
                else:
                    ssim = criterion_ssim(restored, ref)
                    # psnr = criterion_psnr(restored, ref)
                    loss = 1 - ssim
                    loss.backward()
                    scaled_loss += loss.item()
                    # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                    optimizer.step()
                    functional.reset_net(model_restoration)
                torch.cuda.synchronize()
                epoch_loss += loss.item()
                iter       += 1
                for res, tar in zip(restored, ref):
                    train_psnr_val_rgb.append(utils.torchPSNR(res, tar))
                psnr_train = torch.stack(train_psnr_val_rgb).mean().item()
                ssim_train = ssim.item()
                
                writer.add_scalar("loss/iter_loss",  loss.item(), iter)
                writer.add_scalar("loss/epoch_loss", epoch_loss, epoch)
                writer.add_scalar("lr/epoch_loss",   scheduler.get_lr()[0], epoch)
                
            # Evaluation
            if epoch % 1 == 0:
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate(val_loader):
                    image = data_val[0].to(device)
                    ref   = data_val[1].to(device)
                    
                    with torch.no_grad():
                        restored = model_restoration(image)
                    functional.reset_net(model_restoration)
                    
                    for res, tar in zip(restored, ref):
                        psnr_val_rgb.append(utils.torchPSNR(res, tar))

                psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
                ssim_val_rgb = criterion_ssim(restored, ref).item()
                writer.add_scalar("val/psnr", psnr_val_rgb, epoch)
                writer.add_scalar("val/ssim", ssim_val_rgb, epoch)
                if psnr_val_rgb > best_psnr:
                    best_psnr       = psnr_val_rgb
                    best_psnr_epoch = epoch
                    torch.save(model_restoration.state_dict(), str(weights_dir / f"{fullname}_best_psnr.pt"))
                if ssim_val_rgb > best_ssim:
                    best_ssim       = ssim_val_rgb
                    best_ssim_epoch = epoch
                    torch.save(model_restoration.state_dict(), str(weights_dir / f"{fullname}_best_ssim.pt"))
                print("[Epoch %d Training PSNR: %.4f --- best_psnr_epoch %d Test_PSNR %.4f]" % (epoch, psnr_train, best_psnr_epoch, best_psnr))
                print("[Epoch %d Training SSIM: %.4f --- best_ssim_epoch %d Test_SSIM %.4f]" % (epoch, ssim_train, best_ssim_epoch, best_ssim))
            
            # Save model
            if epoch % 50 == 0:
                torch.save(
                    {
                        "epoch"     : epoch,
                        "state_dict": model_restoration.state_dict(),
                        "optimizer" : optimizer.state_dict()
                    },
                    str(weights_dir / f"{fullname}_epoch_{epoch}.pt")
                )
            torch.save(
                {
                    "epoch"     : epoch,
                    "state_dict": model_restoration.state_dict(),
                    "optimizer" : optimizer.state_dict()
                },
                str(weights_dir / f"{fullname}_last.pt")
            )
            scheduler.step()
            print("-" * 150)
            print(
                "Epoch: {}\t"
                "Time: {:.4f}\t"
                "Loss: {:.4f}\t"
                "Train PSNR: {:.4f}\t"
                "SSIM: {:.4f}\t"
                "Learning Rate: {:.8f}\t"
                "Test PSNR: {:.4f}\t"
                "Test SSIM: {:.4f}".format(
                    epoch,
                    time.time() - epoch_start_time,
                    loss.item(),
                    psnr_train,
                    ssim,
                    scheduler.get_lr()[0],
                    best_psnr,
                    best_ssim,
                )
            )
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
