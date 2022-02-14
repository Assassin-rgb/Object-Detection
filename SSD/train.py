from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import pickle
import math
import copy


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class arguments_custom():
  def __init__(self):
    self.description = 'Single Shot MultiBox Detector Training With Pytorch'
    self.input = 300
    self.dataset = 'VOC'
    self.num_class = 21
    self.dataset_root = 'VOCdevkit'
    self.basenet = 'vgg16_reducedfc.pth'
    self.num_epoch = 200
    self.save_folder = 'weights/'
    self.visdom = False
    self.gamma = 0.9
    self.weight_decay = 1e-5
    self.momentum = 0.9
    self.lr = 1e-3
    self.cuda = True
    self.num_workers = 8
    self.start_epoch = 0
    self.resume = False
    self.batch_size = 64

args = arguments_custom()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    cfg = voc
    dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(args.input,
                                                         MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    if args.basenet == 'vgg16_reducedfc.pth':
        ssd_net = build_ssd('train', args.input, args.num_class)
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        state_dict = torch.load(args.resume)
        net.load_state_dict(state_dict)

    else:
        if args.basenet == 'vgg16_reducedfc.pth':
            vgg_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network weights from %s\n'%(args.save_folder + args.basenet))
            ssd_net.vgg.load_state_dict(vgg_weights)


    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method

        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    criterion = MultiBoxLoss(args.num_class, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = 1
    loss_total = []
    loss_loc = []
    loss_cls = []
    print('Loading the dataset...')

    epoch_size = math.ceil(len(dataset) / args.batch_size)
    print('iteration per epoch:', epoch_size)
    print('Training SSD on:', dataset.name)

    step_index = 0
    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # create batch iterator
    # batch_iterator = iter(data_loader)
    for epoch in range(args.start_epoch, args.num_epoch):
        print('\n'+'-'*70+'Epoch: {}'.format(epoch)+'-'*70+'\n')
        if args.visdom and epoch != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1
        if epoch != 0 and epoch % 20 == 0:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
        if epoch <= 5:
            warmup_learning_rate(optimizer,epoch)
        for images, targets in data_loader: # load train data
            # if iteration % 100 == 0:
            for param in optimizer.param_groups:
                if 'lr' in param.keys():
                    cur_lr = param['lr']
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]
            # forward
            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if iteration % 10 == 0:
                print('Epoch '+repr(epoch)+'|| iter ' + repr(iteration % epoch_size)+'/'+repr(epoch_size) +'|| Total iter '+repr(iteration)+ ' || Total Loss: %.4f || Loc Loss: %.4f || Cls Loss: %.4f || LR: %f || timer: %.4f sec.\n' % (loss.item(),loss_l.item(),loss_c.item(),cur_lr,(t1 - t0)), end=' ')
                loss_cls.append(loss_c.item())
                loss_loc.append(loss_l.item())
                loss_total.append(loss.item())
                loss_dic = {'loss':loss_total, 'loss_cls':loss_cls, 'loss_loc':loss_loc}

            if args.visdom:
                update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                                iter_plot, epoch_plot, 'append')

            if iteration != 0 and iteration % 1000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), 'weights/ssd{}_VOC_'.format(args.input) +
                           repr(iteration) + '.pth')
                with open('loss.pkl', 'wb') as f:
                    pickle.dump(loss_dic, f, pickle.HIGHEST_PROTOCOL)
            iteration += 1
    torch.save(net.state_dict(),
               args.save_folder + '' + args.dataset + 'ssd300.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    print('Now we change lr ...')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer,epoch):
    lr_ini = 0.0001
    print('lr warmup...')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_ini+(args.lr - lr_ini)*epoch/5

def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
