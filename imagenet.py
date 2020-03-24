from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import warnings
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

import models.imagenet as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, KLDivLoss
from utils.dataloaders import *
from tensorboardX import SummaryWriter

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data-backend', metavar='BACKEND', default='pytorch',
                    choices=DATA_BACKEND_CHOICES)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

parser.add_argument('--mimic', action='store_true',
                    help='introduce mimicking losses from different branches')
parser.add_argument('--temperature', default=1, type=float,
                    help='temperature for smoothing the soft target')
parser.add_argument('--alpha', default=1, type=float,
                    help='weight of KL divergence loss')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--weight', default='', type=str, metavar='WEIGHT',
                    help='path to pretrained weight (default: none)')

parser.add_argument('--cardinality', type=int, default=32, help='ResNeXt model cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNeXt model base width (number of channels in each group).')
parser.add_argument('--groups', type=int, default=3, help='ShuffleNet model groups')
parser.add_argument('--extent', type=int, default=0, help='GENet model spatial extent ratio')
parser.add_argument('--theta', dest='theta', action='store_true', help='GENet model parameterising the gather function')
parser.add_argument('--excite', dest='excite', action='store_true', help='GENet model combining the excite operator')


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('resnext'):
            model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
        elif args.arch.startswith('shufflenetv1'):
            model = models.__dict__[args.arch](
                    groups=args.groups
                )
        elif args.arch.startswith('ge_resnet'):
            model = models.__dict__[args.arch](
                    extent=args.extent,
                    theta=args.theta,
                    excite=args.excite
                )
        else:
            model = models.__dict__[args.arch]()

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
    title = 'ImageNet-' + args.arch
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
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == 'dali-gpu':
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == 'dali-cpu':
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()

    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size, workers=args.workers)
    val_loader, val_loader_len = get_val_loader(args.data, args.batch_size, workers=args.workers)

    if args.evaluate:
        from collections import OrderedDict
        if os.path.isfile(args.weight):
            print("=> loading pretrained weight '{}'".format(args.weight))
            source_state = torch.load(args.weight)
            target_state = OrderedDict()
            for k, v in source_state.items():
                if k[:7] != 'module.':
                    k = 'module.' + k
                target_state[k] = v
            model.load_state_dict(target_state)
        else:
            print("=> no weight found at '{}'".format(args.weight))
        validate(val_loader, val_loader_len, model, criterion)
        return

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))

        if not args.mimic:
            # train for one epoch
            train_loss, train_loss_head1, train_loss_head2, train_acc, train_acc_head1, train_acc_head2 = train(train_loader, train_loader_len, model, criterion, optimizer, epoch)
            
            # evaluate on validation set
            val_loss, val_loss_head1, val_loss_head2, prec1, prec1_head1, prec1_head2 = validate(val_loader, val_loader_len, model, criterion)

        else:
            # FIXME hard code to parse the outputs
            # train for one epoch
            train_loss, train_loss_head1, train_loss_head2, train_loss_t1_s0, train_loss_t2_s0, train_loss_t0_s1, train_loss_t2_s1, train_loss_t0_s2, train_loss_t1_s2, train_acc, train_acc_head1, train_acc_head2 = train(train_loader, train_loader_len, model, criterion, optimizer, epoch)

            # evaluate on validation set
            val_loss, val_loss_head1, val_loss_head2, val_loss_t1_s0, val_loss_t2_s0, val_loss_t0_s1, val_loss_t2_s1, val_loss_t0_s2, val_loss_t1_s2, prec1, prec1_head1, prec1_head2 = validate(val_loader, val_loader_len, model, criterion)

        lr = optimizer.param_groups[0]['lr']

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


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch):
    bar = Bar('Processing', max=train_loader_len)

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
        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

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

        # FIXME hard code to plot the losses and accuracy
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}, {loss_head1:.4f}, {loss_head2:.4f} | top1: {top1: .4f}, {top1_head1: .4f}, {top1_head2: .4f} | top5: {top5: .4f}, {top5_head1: .4f}, {top5_head2: .4f}'.format(
                    batch=i + 1,
                    size=train_loader_len,
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


def validate(val_loader, val_loader_len, model, criterion):
    bar = Bar('Processing', max=val_loader_len)

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

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        with torch.no_grad():
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
                    size=val_loader_len,
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


from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        if epoch in args.schedule:
            lr *= args.gamma
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
