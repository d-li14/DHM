from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, KLDivLoss
from tensorboardX import SummaryWriter

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', action='store_true',
                    help='set low learning rate to warm up the training')
parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--dropout-rate', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--mimic', action='store_true',
                    help='introduce mimicking losses from different branches')
parser.add_argument('--temperature', default=1, type=float,
                    help='temperature for smoothing the soft target')
parser.add_argument('--alpha', default=1, type=float,
                    help='weight of KL divergence loss')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group) for ResNeXt.')
parser.add_argument('--widening-factor', type=int, default=4, help='Widening factor. 4 -> 64, 8 -> 128, ... for Wide ResNet')
parser.add_argument('--growth-rate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compression-factor', type=float, default=0.5, help='Compression factor (theta) for DenseNet.')
# Miscs
parser.add_argument('--manual-seed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # Validate dataset
    assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    # Use CUDA
    use_cuda = torch.cuda.is_available()

    # Random seed
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manual_seed)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=num_classes)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = 'CIFAR-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                     std=[0.24703233, 0.24348505, 0.26158768])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]) 
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = dataloader(root='./data', train=False, download=False, transform=transform_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))

        if not args.mimic:
            # train for one epoch
            train_loss, train_loss_head1, train_loss_head2, train_acc, train_acc_head1, train_acc_head2 = train(train_loader, model, criterion, optimizer, epoch)
            
            # evaluate on validation set
            val_loss, val_loss_head1, val_loss_head2, prec1, prec1_head1, prec1_head2 = validate(val_loader, model, criterion)

        else:
            # train for one epoch
            train_loss, train_loss_t1_s0, train_loss_t2_s0, train_loss_head1, train_loss_t0_s1, train_loss_t2_s1, train_loss_head2, train_loss_t0_s2, train_loss_t1_s2, train_acc, train_acc_head1, train_acc_head2 = train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            val_loss, val_loss_t1_s0, val_loss_t2_s0, val_loss_head1, val_loss_t0_s1, val_loss_t2_s1, val_loss_head2, val_loss_t0_s2, val_loss_t1_s2, prec1, prec1_head1, prec1_head2 = validate(val_loader, model, criterion)

        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss/head 0', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('loss/head 1', {'train loss': train_loss_head1, 'validation loss': val_loss_head1}, epoch + 1)
        writer.add_scalars('loss/head 2', {'train loss': train_loss_head2, 'validation loss': val_loss_head2}, epoch + 1)
        writer.add_scalars('accuracy/head 0', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)
        writer.add_scalars('accuracy/head 1', {'train accuracy': train_acc_head1, 'validation accuracy': prec1_head1}, epoch + 1)
        writer.add_scalars('accuracy/head 2', {'train accuracy': train_acc_head2, 'validation accuracy': prec1_head2}, epoch + 1)
        #for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)

        if args.mimic:
            writer.add_scalars('loss/teacher1 student0', {'train loss': train_loss_t1_s0, 'validation loss': val_loss_t1_s0}, epoch + 1)
            writer.add_scalars('loss/teacher2 student0', {'train loss': train_loss_t2_s0, 'validation loss': val_loss_t2_s0}, epoch + 1)
            writer.add_scalars('loss/teacher0 student1', {'train loss': train_loss_t0_s1, 'validation loss': val_loss_t0_s1}, epoch + 1)
            writer.add_scalars('loss/teacher2 student1', {'train loss': train_loss_t2_s1, 'validation loss': val_loss_t2_s1}, epoch + 1)
            writer.add_scalars('loss/teacher0 student2', {'train loss': train_loss_t0_s2, 'validation loss': val_loss_t0_s2}, epoch + 1)
            writer.add_scalars('loss/teacher1 student2', {'train loss': train_loss_t1_s2, 'validation loss': val_loss_t1_s2}, epoch + 1)


        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()

    print('Best accuracy:')
    print(best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    bar = Bar('Processing', max=len(train_loader))

    heads = 3
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in range(heads)]
    if args.mimic:
        for i in range(heads * (heads - 1)):
            losses.append(AverageMeter())
    top1 = [AverageMeter() for i in range(heads)]
    top5 = [AverageMeter() for i in range(heads)]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        outputs = model(input)
        ce_loss = []
        kl_loss = []
        for j, output in enumerate(outputs):
            ce_loss.append(criterion(output, target))
            if args.mimic:
                for output_counterpart in outputs:
                    if output_counterpart is not output:
                        kl_loss.append(args.alpha * KLDivLoss(output, output_counterpart.detach(), args.temperature))
                    else:
                        pass
        # indices = np.random.choice(len(kl_loss), int(np.random.rand() * len(kl_loss)))
        # for index in np.unique(indices):
        #     kl_loss[index].detach_()
        loss = ce_loss + kl_loss
        loss_sum = sum(loss)

        # measure accuracy and record loss
        for j in range(len(loss)):
            losses[j].update(loss[j].item(), input.size(0))

        prec1, prec5 = [0.] * heads, [0.] * heads
        for j, output in enumerate(outputs):
            prec1[j], prec5[j] = accuracy(output, target, topk=(1, 5))
            top1[j].update(prec1[j].item(), input.size(0))
            top5[j].update(prec5[j].item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}, {loss_head1:.4f}, {loss_head2:.4f} | top1: {top1: .4f}, {top1_head1: .4f}, {top1_head2: .4f} | top5: {top5: .4f}, {top5_head1: .4f}, {top5_head2: .4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses[0].avg,
                    loss_head1=losses[1].avg,
                    loss_head2=losses[2].avg,
                    top1=top1[0].avg,
                    top1_head1=top1[1].avg,
                    top1_head2=top1[2].avg,
                    top5=top5[0].avg,
                    top5_head1=top5[1].avg,
                    top5_head2=top5[2].avg,
                    )
        bar.next()
    bar.finish()
    return [l.avg for l in losses] + [t.avg for t in top1]


def validate(val_loader, model, criterion):
    bar = Bar('Processing', max=len(val_loader))

    heads = 3
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in range(heads)]
    if args.mimic:
        for i in range(heads * (heads - 1)):
            losses.append(AverageMeter())
    top1 = [AverageMeter() for i in range(heads)]
    top5 = [AverageMeter() for i in range(heads)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)

            # compute output
            outputs = model(input)
            ce_loss = []
            kl_loss = []
            for j, output in enumerate(outputs):
                ce_loss.append(criterion(output, target))
                for output_counterpart in outputs:
                    if args.mimic:
                        if output_counterpart is not output:
                            kl_loss.append(args.alpha * KLDivLoss(output, output_counterpart.detach(), args.temperature))
                        else:
                            pass
            loss = ce_loss + kl_loss

            # measure accuracy and record loss
            for j in range(len(loss)):
                losses[j].update(loss[j].item(), input.size(0))

            prec1, prec5 = [0.] * heads, [0.] * heads
            for j, output in enumerate(outputs):
                prec1[j], prec5[j] = accuracy(output, target, topk=(1, 5))
                top1[j].update(prec1[j].item(), input.size(0))
                top5[j].update(prec5[j].item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}, {loss_head1:.4f}, {loss_head2:.4f} | top1: {top1: .4f}, {top1_head1: .4f}, {top1_head2: .4f} | top5: {top5: .4f}, {top5_head1: .4f}, {top5_head2: .4f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses[0].avg,
                    loss_head1=losses[1].avg,
                    loss_head2=losses[2].avg,
                    top1=top1[0].avg,
                    top1_head1=top1[1].avg,
                    top1_head2=top1[2].avg,
                    top5=top5[0].avg,
                    top5_head1=top5[1].avg,
                    top5_head2=top5[2].avg,
                    )
        bar.next()
    bar.finish()
    return [l.avg for l in losses] + [t.avg for t in top1]


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    """Sets the learning rate to the initial LR decayed by 10 following schedule"""
    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** (epoch // args.step))
    elif args.lr_decay == 'schedule':
        if epoch in args.schedule:
            lr *= args.gamma
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if args.warmup:
        if epoch == 0:
            lr = 0.01
        elif epoch == 1:
            lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
