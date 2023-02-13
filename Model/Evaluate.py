# -*- coding: utf-8 -*-
"""
# @file name  : Evaluate.py
# @author     : Siyuan SONG
# @date       : 2023-01-20 12:49:00
# @brief      : CEPC PID
"""
import torch
import numpy as np
from Net import lenet
from Config.config import parser
import matplotlib.pyplot as plt
from Data import loader
import os
from torch.nn import Softmax
# from Train import ckp_dir, MEAN, STD
from ANA.acc import plotACC
from ANA.acc_extra import plotACCExtra
from ANA.distribution import plotDistribution
from ANA.roc import plotROC
from Config.config import parser
from torchmetrics.classification import MulticlassROC, MulticlassAUROC


def totalACC(data_loader, net, device):
    # evaluate
    correct_val = 0.
    total_val = 0.
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # input configuration
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).squeeze().sum().cpu().numpy()
        acc = "{:.2f}".format(100 * correct_val / total_val)
        # print("acc: {}%".format(acc))
        return float(acc)


def pbDisctuibution(data_loader, net, save_path, device):
    distributions = []
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            # input configuration
            inputs = inputs.to(device)

            outputs = net(inputs)
            prbs = Softmax(dim=1)(outputs)
            if j == 0:
                distributions = prbs.cpu().numpy()
            else:
                distributions = np.append(distributions, prbs.cpu().numpy(), axis=0)
        np.save(save_path, distributions)


def getROC(data_loader, net, device, save_path, num_class=3, ignore_index=None):
    preds = torch.tensor([])
    targets = torch.tensor([])
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # input configuration
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            prbs = Softmax(dim=1)(outputs)
            if j == 0:
                preds = prbs
                targets = labels
            else:
                preds = torch.cat((preds, prbs), 0)
                targets = torch.cat((targets, labels), 0)
        metric = MulticlassROC(num_classes=num_class, thresholds=None, ignore_index=ignore_index)
        fprs_, tprs_, thresholds_ = metric(preds, targets)
        fprs = []
        tprs = []
        for i, fpr in enumerate(fprs_):
            fprs.append(fpr.cpu().numpy())
            tprs.append(tprs_[i].cpu().numpy())

        np.array(fprs)
        np.array(tprs)
        np.save(save_path.format('fpr'), fprs)
        np.save(save_path.format('tpr'), tprs)

        mc_auroc = MulticlassAUROC(num_classes=num_class, average=None, thresholds=None, ignore_index=ignore_index)
        auroc = mc_auroc(preds, targets)
        np.save(save_path.format('auroc'), auroc.cpu().numpy())


def get_file_name(path):  # get .pth file
    image_files = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.pth':
            return file
    return None


def evaluate(root_path, mean, std, n_classes):
    # Control flag
    comb_flag = True
    sep_flag = True
    dis_flags = True

    # load model
    root_path = root_path
    MEAN = mean
    STD = std
    model_path = os.path.join(root_path, get_file_name(root_path))

    ana_dir = os.path.join(root_path, 'ANA')
    if not os.path.exists(ana_dir):
        os.mkdir(ana_dir)

    fig_dir = os.path.join(ana_dir, 'Fig')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    save_combin_dir = os.path.join(ana_dir, 'ahcal_beam_test_combination')  # all test set combined
    if not os.path.exists(save_combin_dir):
        os.mkdir(save_combin_dir)
    save_combin_path = os.path.join(save_combin_dir, '{}.npy')  # store accuracy

    save_sep_dir = os.path.join(ana_dir, 'ahcal_beam_test_seperate')  # seperate test set
    if not os.path.exists(save_sep_dir):
        os.mkdir(save_sep_dir)
    save_sep_path = os.path.join(save_sep_dir, '{}.npy')  # store accuracy

    save_extra_dir = os.path.join(ana_dir, 'ahcal_beam_test_seperate_extra')  # extra energy point
    if not os.path.exists(save_extra_dir):
        os.mkdir(save_extra_dir)
    save_extra_path = os.path.join(save_extra_dir, '{}.npy')  # store accuracy

    # combination
    combin_datasets_dir_dict={3:'/lustre/collider/songsiyuan/CEPC/PID/Trainsets/ahcal_testbeam_simu/test',
                              4:'/lustre/collider/songsiyuan/CEPC/PID/Trainsets/ahcal_beam_test_mu_e_pi_proton/test',
                              }
    combin_datasets_dir = combin_datasets_dir_dict.get(n_classes)
    combin_datasets_path = os.path.join(combin_datasets_dir, 'datasets.npy')
    combin_labels_path = os.path.join(combin_datasets_dir, 'labels.npy')

    # seperate energy points
    sep_datasets_dir = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/ahcal_testbeam_seperate_testsets'
    sep_e_pi_proton_datasets_path = os.path.join(sep_datasets_dir, 'ahcal_{}_{}GeV_2cm_10k/imgs.npy')
    sep_e_pi_proton_labels_path = os.path.join(sep_datasets_dir, 'ahcal_{}_{}GeV_2cm_10k/labels.npy')
    sep_e_pi_proton_energy_points = sorted([20, 30, 40, 50, 60, 70, 80, 90, 100, 120])

    sep_mu_datasets_path = os.path.join(sep_datasets_dir, 'ahcal_{}_{}GeV_2cm_10k/imgs.npy')
    sep_mu_labels_path = os.path.join(sep_datasets_dir, 'ahcal_{}_{}GeV_2cm_10k/labels.npy')
    sep_mu_energy_points = sorted([160])

    # extra energy points
    extra_datasets_dir = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/extra_energy_point'
    extra_e_pi_proton_datasets_path = os.path.join(extra_datasets_dir, 'ahcal_{}_{}GeV_2cm_1k/datasets.npy')
    extra_e_pi_proton_labels_path = os.path.join(extra_datasets_dir, 'ahcal_{}_{}GeV_2cm_1k/labels.npy')
    extra_e_pi_proton_energy_points = sorted([15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 110, 115, 125, 130])

    extra_mu_datasets_path = os.path.join(extra_datasets_dir, 'ahcal_{}_{}GeV_2cm_1k/datasets.npy')
    extra_mu_labels_path = os.path.join(extra_datasets_dir, 'ahcal_{}_{}GeV_2cm_1k/labels.npy')
    extra_mu_energy_points = sorted([100, 120, 130, 140, 150, 170, 180, 190, 200])

    #   distribution
    save_dis_dir = os.path.join(ana_dir, 'ahcal_beam_test_dis')
    if not os.path.exists(save_dis_dir):
        os.mkdir(save_dis_dir)
    save_dis_path = os.path.join(save_dis_dir, '{}_dis.npy')

    dis_datasets_path = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/ahcal_beam_test_mu_e_pi_proton/{}/datasets.npy'
    dis_labels_path = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/ahcal_beam_test_mu_e_pi_proton/{}/labels.npy'

    # roc

    save_roc_dir = os.path.join(ana_dir, 'roc')
    if not os.path.exists(save_roc_dir):
        os.mkdir(save_roc_dir)
    save_roc_path = os.path.join(save_roc_dir, '{}.npy')
    fpr_path = save_roc_path.format('fpr')
    tpr_path = save_roc_path.format('tpr')
    auroc_path = save_roc_path.format('auroc')
    bdt_path = '/lustre/collider/songsiyuan/CEPC/PID/BDT/bdt_roc/pion_roc_bdt.txt'

    ####################################################################################

    net = lenet.LeNet_bn(classes=n_classes)
    if torch.cuda.is_available():
        net = net.cuda()
        net.load_state_dict(torch.load(model_path))
        device = 'cuda'
    else:
        device = 'cpu'
        net.load_state_dict(torch.load(model_path, map_location=device))

    #  combination

    if comb_flag:
        # data loader
        img_test_path = os.path.join(root_path, combin_datasets_path)
        label_test_path = os.path.join(root_path, combin_labels_path)
        loader_test = loader.data_loader(img_test_path, label_test_path, mean=MEAN, std=STD,
                                         mean_std_static=True, num_workers=0, batch_size=1000)
        acc = totalACC(loader_test, net, device)
        np.save(save_combin_path.format('combination'), np.array([acc]))

    if sep_flag:
        # caculate seperate acc
        particles_sep_dict={3:['e+', 'pi+'],
                            4:['e+', 'pi+','proton']}
        particles = particles_sep_dict.get(n_classes)
        for particle in particles:
            accs = np.zeros((2, len(sep_e_pi_proton_energy_points)))
            for (j, energy_point) in enumerate(sep_e_pi_proton_energy_points):
                # data loader
                img_test_path = os.path.join(root_path, sep_e_pi_proton_datasets_path).format(particle, energy_point)
                label_test_path = os.path.join(root_path, sep_e_pi_proton_labels_path).format(particle, energy_point)
                loader_test = loader.data_loader(img_test_path, label_test_path, mean=MEAN, std=STD,
                                                 mean_std_static=True, num_workers=0, batch_size=1000)

                acc = totalACC(loader_test, net, device)
                accs[0, j] = energy_point
                accs[1, j] = acc
            np.save(save_sep_path.format(particle), accs)

        particles = ['mu+']
        for particle in particles:
            accs = np.zeros((2, len(sep_mu_energy_points)))
            for j, energy_point in enumerate(sep_mu_energy_points):
                # data loader
                img_test_path = os.path.join(root_path, sep_mu_datasets_path).format(particle, energy_point)
                label_test_path = os.path.join(root_path, sep_mu_labels_path).format(particle, energy_point)
                loader_test = loader.data_loader(img_test_path, label_test_path, mean=MEAN, std=STD,
                                                 mean_std_static=True, num_workers=0, batch_size=1000)
                acc = totalACC(loader_test, net, device)
                accs[0, j] = energy_point
                accs[1, j] = acc
            np.save(save_sep_path.format(particle), accs)

        # caculate seperate extra energy points acc
        particles = particles_sep_dict.get(n_classes)
        for particle in particles:
            accs = np.zeros((2, len(extra_e_pi_proton_energy_points)))
            for (j, energy_point) in enumerate(extra_e_pi_proton_energy_points):
                # data loader
                img_test_path = os.path.join(root_path, extra_e_pi_proton_datasets_path).format(particle, energy_point)
                label_test_path = os.path.join(root_path, extra_e_pi_proton_labels_path).format(particle, energy_point)
                loader_test = loader.data_loader(img_test_path, label_test_path, mean=MEAN, std=STD,
                                                 mean_std_static=True, num_workers=0, batch_size=1000)

                acc = totalACC(loader_test, net, device)
                accs[0, j] = energy_point
                accs[1, j] = acc
            np.save(save_extra_path.format(particle), accs)

        particles = ['mu+']
        for particle in particles:
            accs = np.zeros((2, len(extra_mu_energy_points)))
            for j, energy_point in enumerate(extra_mu_energy_points):
                # data loader
                img_test_path = os.path.join(root_path, extra_mu_datasets_path).format(particle, energy_point)
                label_test_path = os.path.join(root_path, extra_mu_labels_path).format(particle, energy_point)
                loader_test = loader.data_loader(img_test_path, label_test_path, mean=MEAN, std=STD,
                                                 mean_std_static=True, num_workers=0, batch_size=1000)
                acc = totalACC(loader_test, net, device)
                accs[0, j] = energy_point
                accs[1, j] = acc
            np.save(save_extra_path.format(particle), accs)

    # probability distribution
    if dis_flags:
        particles_dict = {3:['mu+', 'e+', 'pi+'],
                          4:['mu+', 'e+', 'pi+', 'proton']}
        particles = particles_dict.get(n_classes)
        for particle in particles:
            img_dis_path = os.path.join(root_path, dis_datasets_path).format(particle)
            label_dis_path = os.path.join(root_path, dis_labels_path).format(particle)
            loader_dis = loader.data_loader(img_dis_path, label_dis_path, mean=MEAN, std=STD,
                                            mean_std_static=True, num_workers=0, batch_size=1000)
            pbDisctuibution(loader_dis, net, save_dis_path.format(particle), device)

    # roc

    img_test_path = os.path.join(root_path, combin_datasets_path)
    label_test_path = os.path.join(root_path, combin_labels_path)
    loader_test = loader.data_loader(img_test_path, label_test_path, mean=MEAN, std=STD,
                                     mean_std_static=True, num_workers=0, batch_size=1000)
    getROC(loader_test, net, device, save_roc_path, n_classes)

    # plot

    # acc
    combin_acc_path = save_combin_path.format('combination')
    pi_path_sep = save_sep_path.format('pi+')
    e_path_sep = save_sep_path.format('e+')
    mu_path_sep = save_sep_path.format('mu+')
    proton_path_sep=save_sep_path.format('proton')
    save_acc_path = os.path.join(fig_dir, 'acc.png')
    plotACC(combi_path=combin_acc_path, mu_path=mu_path_sep, e_path=e_path_sep, pi_path=pi_path_sep, proton_path=proton_path_sep,
            save_path=save_acc_path,n_classes=n_classes)

    # acc with extra energy points
    pi_path_extra = save_extra_path.format('pi+')
    e_path_extra = save_extra_path.format('e+')
    mu_path_extra = save_extra_path.format('mu+')
    proton_path_extra=save_extra_path.format('proton')
    save_acc_extra_path = os.path.join(fig_dir, 'acc_extra.png')
    plotACCExtra(mu_path=mu_path_sep, e_path=e_path_sep, pi_path=pi_path_sep, proton_path=proton_path_sep,
                 mu_extra_path=mu_path_extra, e_extra_path=e_path_extra, pi_extra_path=pi_path_extra, proton_extra_path=proton_path_extra,
                 save_path=save_acc_extra_path, n_classes=n_classes)

    # probability distribution
    pi_path_dis = save_dis_path.format('pi+')
    e_path_dis = save_dis_path.format('e+')
    mu_path_dis = save_dis_path.format('mu+')
    proton_path_dis = save_dis_path.format('proton')
    save_dis_compare_path = os.path.join(fig_dir, '{}_dis{}{}.png')

    for log in [True, False]:
        for stack in [True, False]:
            plotDistribution(mu_path=mu_path_dis, e_path=e_path_dis, pi_path=pi_path_dis, proton_path=proton_path_dis,
                             log=log, stack=stack,save_path=save_dis_compare_path, n_classes=n_classes)

    # roc
    save_roc_path = os.path.join(fig_dir, 'ann_bdt_compare.png')
    plotROC(fpr_path=fpr_path, tpr_path=tpr_path, auroc_path=auroc_path, signal='pi+', bdt_path=bdt_path,compare=False,
            save_path=save_roc_path)


if __name__ == '__main__':
    args = parser.parse_args()
    # load model
    root_path = '/lustre/collider/songsiyuan/CEPC/PID/CheckPoint/epoch_300'
    N_CLASSES = 3
    mean=args.mean
    std=args.std
    evaluate(root_path=root_path, n_classes=N_CLASSES,mean=mean,std=std)
