import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import time
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from config import get_config
from models.retina import RetinaFace
from utils.multibox_loss import MultiBoxLoss
from utils.prior_box import PriorBox
from utils.transform import Augmentation

from dataset.dataset import WiderFaceDataset
from argparse import ArgumentParser

from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(description = 'Arguments For RetinaFace Training Process')

    parser.add_argument('--train-data', type=str, default='data/widerface', help='Train data path')
    parser.add_argument('--backbone', type=str, default='resnet34', choices=['resnet34', 'resnet50'], help='Model backbone')
    parser.add_argument('--num-workers', type=int, default=8, help='Num workers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

    # Training
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--print-freq', type=int, default=10, help='Print frequency during training')

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr-warmup-epochs', type=int, default=100, help='Learning rate warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--power', type=float, default=0.9, help='Power for learning rate policy')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma update for SGD')

    parser.add_argument('--save-dir', type=str, default='checkpoint', help='Save directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume training from last checkpoint')
    parser.add_argument('--tensorboard', type=str, default='tensorboard', help='Tensorboard')

    args = parser.parse_args()
    return args

rgb_mean = (104, 117, 123) # BGR order


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, criterion, optimizer, data_loader, epoch, device, print_freq=1, scaler=None, writer=None):
    model.train()
    batch_loss = []
    loc_losses = []
    conf_losses = []
    land_losses = []
    progress_bar = tqdm(data_loader, colour='green', total=len(data_loader), dynamic_ncols=True, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
    for batch_idx, (images, targets) in enumerate(progress_bar):
        start_time = time.time()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(images)
            loss_loc, loss_conf, loss_land = criterion(outputs, targets)
            loss = cfg['loc_weight'] * loss_loc + loss_conf + loss_land

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # GLOBAL STEP
        global_step = epoch * len(data_loader) + batch_idx

        lr = optimizer.param_groups[0]['lr']
        # TENSORBOARD BATCH LOG
        if writer is not None:
            writer.add_scalar("Loss_batch/total", loss.item(), global_step)
            writer.add_scalar("Loss_batch/localization", loss_loc.item(), global_step)
            writer.add_scalar("Loss_batch/confidence", loss_conf.item(), global_step)
            writer.add_scalar("Loss_batch/landmarks", loss_land.item(), global_step)
            writer.add_scalar('Learning Rate', lr, global_step)

        if (batch_idx + 1) % print_freq == 0:
            # lr = optimizer.param_groups[0]['lr']
            # print(
            #     f"Epoch: {epoch + 1}/{cfg['epochs']} | Batch: {batch_idx + 1}/{len(data_loader)} | "
            #     f"Loss Localization : {loss_loc.item():.4f} | Classification: {loss_conf.item():.4f} | "
            #     f"Landmarks: {loss_land.item():.4f} | "
            #     f"LR: {lr:.8f} | Time: {(time.time() - start_time):.4f} s"
            # )
            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                "loc": f"{loss_loc.item():.4f}",
                "conf": f"{loss_conf.item():.4f}",
                "land": f"{loss_land.item():.4f}",
                "lr": f"{lr:.6f}",
                "time": f"{(time.time() - start_time):.3f}s"
            })

        batch_loss.append(loss.item())
        loc_losses.append(loss_loc.item())
        conf_losses.append(loss_conf.item())
        land_losses.append(loss_land.item())

    avg_loss = np.mean(batch_loss)
    avg_loc_loss = np.mean(loc_losses)
    avg_conf_loss = np.mean(conf_losses)
    avg_land = np.mean(land_losses)

    print(f"Average batch loss: {avg_loss:.7f}")
    if writer is not None:
        writer.add_scalar("Loss_epoch/total", avg_loss, epoch)
        writer.add_scalar("Loss_epoch/localization", avg_loc_loss, epoch)
        writer.add_scalar("Loss_epoch/confidence", avg_conf_loss, epoch)
        writer.add_scalar("Loss_epoch/landmarks", avg_land, epoch)

def main(args):
    random_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.tensorboard)

    train_augmentation = Augmentation(cfg['image_size'], rgb_mean)
    train_dataset = WiderFaceDataset(args.train_data, train=True, transform=train_augmentation)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )
    print("Successfully load train dataset")

    # Generate prior boxes
    prior_box = PriorBox(cfg, image_size=(cfg['image_size'], cfg['image_size']))
    priors = prior_box.generate_anchors()
    priors = priors.to(device)

    # Multibox Loss
    criterion = MultiBoxLoss(priors=priors, threshold=0.35, neg_pos_ratio=7, variance=cfg['variance'], device=device)

    # Model
    model = RetinaFace(cfg=cfg).to(device)

    # Optimizer
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=args.gamma)

    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(f'{args.checkpoint}/{args.backbone}_checkpoint.ckpt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Successfully load {args.checkpoint}/{args.backbone}_checkpoint.ckpt")

    print("TRAINING!!!")
    for epoch in range(start_epoch, cfg['epochs']):
        train_one_epoch(model, criterion, optimizer, train_dataloader, epoch, device, args.print_freq, scaler=None, writer=writer)
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }
        lr_scheduler.step()

        torch.save(ckpt, f'{args.save_dir}/{args.backbone}_checkpoint.ckpt')
        torch.save(model.state_dict(), f'{args.save_dir}/{args.backbone}_last.ckpt')

    # Final model
    state = model.state_dict()
    torch.save(state, f'{args.save_dir}/{args.backbone}_final.ckpt')

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.backbone)
    main(args)