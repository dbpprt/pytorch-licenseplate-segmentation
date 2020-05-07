import abc
import datetime
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np

import cv2
import lovasz_losses as L
import torch
import torch.utils.data
import torchvision
import transforms as T
import utils
from dataloader import SegmentationDataset
from model import create_model
from swa import SWA
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage


def get_dataset(image_set, transform, dataset_dir):
    return SegmentationDataset(folder_path=os.path.join(dataset_dir, image_set), transforms=transform)


def get_transform(train):
    base_size = 520
    crop_size = 480

    min_size = int((0.5 if train else 1.0) * base_size)
    max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    transforms.append(T.RandomResize(min_size, max_size))
    if train:
        transforms.append(T.RandomColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25))
        transforms.append(T.RandomGaussianSmoothing(radius=[0, 5]))
        transforms.append(T.RandomRotation(degrees=30, fill=0))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomPerspective(fill=0))
        transforms.append(T.RandomCrop(crop_size, fill=0))
        transforms.append(T.RandomGrayscale(p=0.1))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def evaluate(model, data_loader, device, epoch = None, writer = None, print_freq = 1):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Test:'

    iou_list = []

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image, target = image.to(device), target.to(device)

            output = model(image)

            loss, iou = criterion(output, target)

            iou_list.append(iou)
            metric_logger.update(
                loss=loss.item(), iou=iou, mIOU=np.mean(iou_list))

            if writer is not None:
                writer.add_scalar('loss/test', loss.item(), epoch)
                writer.add_scalar('iou/test', iou, epoch)

    if writer is not None:
        writer.add_scalar('miou/test', np.mean(iou_list), epoch)

    print(f'{header} mIOU: {np.mean(iou_list)}')


def criterion(inputs, target):
    sigmoid = torch.sigmoid(inputs['out'])
    sigmoid_aux = torch.sigmoid(inputs['aux'])
    preds = (inputs['out'].data > 0).long()

    loss = L.lovasz_softmax(sigmoid, target, classes=[1], ignore=128)
    loss_aux = L.lovasz_softmax(sigmoid_aux, target, classes=[1], ignore=128)

    iou = L.iou_binary(preds, target, ignore=128, per_image=True)

    return loss + 0.5 * loss_aux, iou


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, writer, print_freq, use_swa):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        output = model(image)

        loss, iou = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"], iou=iou)

        if random.random() < 0.15:
            writer.add_image(
                'input/train', torchvision.utils.make_grid([torchvision.utils.make_grid(image), torchvision.utils.make_grid(target), torchvision.utils.make_grid(output['out'].data, normalize=True)], nrow=1), epoch)

        writer.add_scalar('loss/train', loss.item(), epoch)
        writer.add_scalar('lr/train', optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar('iou/train', iou, epoch)

    if use_swa:
        optimizer.swap_swa_sgd()


def main(args):
    torch.cuda.empty_cache()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    device = torch.device(args.device)

    dataset = get_dataset("train", get_transform(train=True), args.dataset_dir)
    dataset_test = get_dataset("val", get_transform(train=False), args.dataset_dir)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    model = create_model(aux_loss=True, freeze_backbone=args.freeze_backbone)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    if args.test_only:
        evaluate(model, data_loader_test,
                 device=device, print_freq=1)
        return

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters()
                    if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters()
                    if p.requires_grad]},
        {"params": [p for p in model.aux_classifier.parameters()
                    if p.requires_grad]},
    ]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.use_swa:
        optimizer = SWA(optimizer, swa_start=args.swa_start, swa_freq=args.swa_freq)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    start_time = time.time()
    writer = SummaryWriter()

    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader,
                        lr_scheduler, device, epoch, writer, args.print_freq, args.use_swa)
        evaluate(model, data_loader_test,
                 device, epoch, writer, print_freq=args.print_freq)

        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args
            },
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    writer.close()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='PyTorch Segmentation Training')

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--swa-start', default=10, type=int,
                        help='number of steps before starting to apply SWA (default 10)',
                        dest='swa_start')
    parser.add_argument('--swa-freq', default=5, type=int,
                        help='number of steps between subsequent updates of SWA',
                        dest='swa_freq')
    parser.add_argument('--print-freq', default=10,
                        type=int, help='print frequency')
    parser.add_argument('--dataset-dir', default='./dataset', help='dataset path')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        help="Freeze backbone",
        action="store_true",
    )
    parser.add_argument(
        "--swa",
        dest="use_swa",
        help="Use Stochastic Weight Averaging",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
