# -*- coding: utf-8 -*-
"""
# @file name  : Train.py
# @author     : Siyuan SONG
# @date       : 2023-01-20 15:09:00
# @brief      : CEPC PID
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
# Evaluate
from torch.nn import Softmax
from ANA.acc import plotACC
from ANA.acc_extra import plotACCExtra
from ANA.distribution import plotDistribution
from Evaluate import evaluate

import Net.lenet
from Config.config import parser
from Data import loader
import sys

hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".." + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

from Net import lenet
from Data import loader
import SetSeed

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

args = parser.parse_args()
if args.set_seed:
    SetSeed.setupSeed(args.seed)  # set random seed

# set hyper-parameters
MAX_EPOCH = args.n_epoch
BATCH_SIZE = args.batch_size
LR = args.learning_rate
log_interval = args.log_interval
val_interval = args.val_interval
NUM_WORKERS = args.num_workers
MEAN = args.mean
STD = args.std
OPTIM = args.optim
N_CLASSES=args.n_classes

# path
# TODO
data_dir_dict={4:'/ahcal_beam_test_mu_e_pi_proton',
           3:'ahcal_beam_test_mu_e_pi_no_energy',
          }
net_name = 'epoch_{}_lr_{}_batch{}_optim_{}_classes_{}_no_energy'.format(MAX_EPOCH, LR, BATCH_SIZE, OPTIM, N_CLASSES)

root_path = '/home/songsiyuan/CEPC/PID/Model'  # train.py's dir
data_path = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets'
data_dir=data_dir_dict.get(N_CLASSES) #Datasets dir
ckp_dir = os.path.join('/lustre/collider/songsiyuan/CEPC/PID/CheckPoint/', net_name)
if not os.path.exists(ckp_dir):
    os.mkdir(ckp_dir)
model_path = os.path.join(ckp_dir, 'net.pth')
loss_path = ckp_dir + '/loss.png'
par_path = ckp_dir + '/hyper_paras.txt'

if __name__ == '__main__':
    # save hyper-parameters
    dict = {'MAX_EPOCH': MAX_EPOCH, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR, 'MEAN': MEAN, 'STD': STD, 'OPTIM': OPTIM
        , 'N_CLASSES':N_CLASSES,}

    filename = open(par_path, 'w')  # dict to txt
    for k, v in dict.items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()
    # ============================ step 1/5 data ============================

    # DataLoder
    img_train_path = data_path + data_dir+'/train/datasets.npy'
    label_train_path = data_path + data_dir + '/train/labels.npy'

    img_valid_path = data_path + data_dir + '/validation/datasets.npy'
    label_valid_path = data_path + data_dir + '/validation/labels.npy'

    img_test_path = data_path + data_dir + '/test/datasets.npy'
    label_test_path = data_path + data_dir + '/test/labels.npy'

    loader_train = loader.data_loader(img_train_path, label_train_path, mean=MEAN, std=STD,
                                      num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
    loader_valid = loader.data_loader(img_valid_path, label_valid_path, mean=MEAN, std=STD,
                                      num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
    # loader_test = loader.data_loader(img_test_path, label_test_path, num_workers=NUM_WORKERS)
    # ============================ step 2/5 model ============================

    net = Net.lenet.LeNet_bn(classes=N_CLASSES)
    net.initialize_weights()

    # ============================ step 3/5 loss function ============================
    criterion = nn.CrossEntropyLoss()

    # ============================ step 4/5 optimizer ============================
    if OPTIM == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if OPTIM == 'Adam':
        optimizer = optim.AdamW(net.parameters(), lr=LR, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ============================ step 5/5 train ============================

    if args.gpu:
        net = net.cuda()
        criterion = criterion.cuda()

        # gpu= os.environ["CUDA_VISIBLE_DEVICES"]
        device = torch.device('cuda')
    else:
        device = "cpu"

    train_curve = list()
    valid_curve = list()

    for epoch in range(MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, (inputs, labels) in enumerate(loader_train):
            if args.gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # input configuration
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # analyze results
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().cpu().numpy()

            # print results
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i + 1, len(loader_train), loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step()  # renew LR

        # validate the model
        if (epoch + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, (inputs, labels) in enumerate(loader_valid):
                    if args.gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # input configuration
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                    loss_val += loss.item()

                loss_val_epoch = loss_val / len(loader_valid)
                valid_curve.append(loss_val_epoch)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j + 1, len(loader_valid), loss_val_epoch, correct_val / total_val))

            # save CKP
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': net.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': loss,}, ck_path)

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(loader_train)
    valid_x = np.arange(1,
                        len(valid_curve) + 1) * train_iters * val_interval - 1  # valid records epochloss，need to be converted to iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.savefig(loss_path)


    # save model
    torch.save(net.state_dict(), model_path)

    # ============================ evaluate model ============================

    evaluate(root_path=ckp_dir, mean=MEAN, std=STD, n_classes=N_CLASSES)