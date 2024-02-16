# -*- coding: utf-8 -*-
"""
# @file name  : Evaluate.py
# @author     : Siyuan SONG
# @date       : 2023-01-20 12:49:00
# @brief      : CEPC PID
"""
import torch
import numpy as np
import pandas as pd
from Config.config import parser
import matplotlib.pyplot as plt
from Data import loader
import os
from torch.nn import Softmax
from ANA.acc import plotACC, plotACCbar, plot_purity_threshold, plot_purity_ep
from ANA.distribution import plotDistribution
from ANA.roc import plotROC, plot_s_b_threshold, plot_s_b_ep, plot_s_b_ratio_threshold, calculate_auc
from ANA.ann_ana import ANN_ANA
from Config.config import parser
from torchmetrics.classification import MulticlassROC, MulticlassAUROC, MulticlassAccuracy
from Net.lenet import LeNet_bn
from Net.resnet import ResNet, BasicBlock, Bottleneck, ResNet_Avg
from PID import npyPID, pid_data_loader
from Data.loader import data_loader
import copy
import torch
import uproot
import glob


class ReadRoot():

    def __init__(self, file_path, tree_name, start=None, end=None, cut=None, exp=None):
        file = uproot.open(file_path)
        tree = file[tree_name]

        self.tree = tree.arrays(aliases=None, cut=cut, expressions=exp, library="np", entry_start=start,
                                entry_stop=end)

    def readBranch(self, branch):
        return self.tree[branch]


def purity_at_thresholds(model, dataloader, device, num_classes, thresholds_num=100):
    tps = np.zeros((num_classes, thresholds_num))
    nums = np.zeros((num_classes, thresholds_num))
    purities = np.zeros((num_classes, thresholds_num))

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = Softmax(dim=1)(outputs)

            values, predicted = torch.max(outputs, 1)

            for t in range(thresholds_num):
                threshold = (t + 1) / float(thresholds_num)
                cut = values > threshold
                valid_preds = predicted[cut]
                valid_labels = labels[cut]
                for c in range(num_classes):
                    tps[c, t] += ((valid_preds == c) & (valid_labels == c)).cpu().float().sum().item()
                    nums[c, t] += (valid_preds == c).cpu().float().sum().item()

    for c in range(num_classes):
        for t in range(thresholds_num):
            purities[c, t] = tps[c, t] / nums[c, t] if nums[c, t] != 0 else 0

    # print(purities)
    return purities


def totalACC(data_loader, net, device):
    # evaluate
    correct_val = 0.
    total_val = 0.
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):
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


def ACCParticle(data_loader, net, device, n_classes, threshold=0.9):
    # evaluate
    correct_val = np.zeros(n_classes)
    total_val = np.zeros(n_classes)

    predicts = []
    targets = []
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):
            # input configuration
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            # _, predicted = torch.max(outputs.data, 1)
            #
            # for type in range(n_classes):
            #
            #     total_val[type] += (labels[labels==type]).size(0)
            #     correct_val[type] += (predicted[labels==type] == labels[labels==type]).squeeze().sum().cpu().numpy()

            # acc = 100 * correct_val / total_val

            predicts.append(outputs)
            targets.append(labels)
        targets = torch.cat(targets)
        predicts = torch.cat(predicts)

        mca = MulticlassAccuracy(num_classes=n_classes, average=None, threshold=threshold).to(device)
        acc = 100 * mca(predicts, targets).cpu().numpy()
        # print("acc: {}%".format(acc))
        return acc


def pbDisctuibution(data_loader, net, save_path, device):
    distributions = []
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):

            # input configuration
            inputs = inputs.to(device)

            outputs = net(inputs)
            prbs = Softmax(dim=1)(outputs)
            if j == 0:
                distributions = prbs.cpu().numpy()
            else:
                distributions = np.append(distributions, prbs.cpu().numpy(), axis=0)
        np.save(save_path, distributions)


def getROC(data_loader, net, device, save_path, num_class, ignore_index=None, threshold_num=21):
    preds = torch.tensor([])
    targets = torch.tensor([])
    with torch.no_grad():
        net.eval()
        for j, (inputs, labels) in enumerate(data_loader):

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
        metric = MulticlassROC(num_classes=num_class, thresholds=threshold_num, ignore_index=ignore_index).to(device)
        fprs_, tprs_, thresholds_ = metric(preds, targets)
        fprs = []
        tprs = []
        for i, fpr in enumerate(fprs_):
            fprs.append(fpr.cpu().numpy())
            tprs.append(tprs_[i].cpu().numpy())

        np.array(fprs, dtype=object)
        np.array(tprs, dtype=object)
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


def evaluate(root_path,
             n_classes,
             net,
             data_loader_func,
             combin_datasets_dir_dict,
             sep_datasets_dir_dict,
             data_type,
             fig_dir_name='Fig',
             threshold=0.9,
             threshold_num=21,
             comb_flag=True,
             sep_flag=False,
             dis_flags=False,
             max_nodes=512):
    # load model

    model_path = os.path.join(root_path, get_file_name(root_path))

    ana_dir = os.path.join(root_path, 'ANA')
    if not os.path.exists(ana_dir):
        os.mkdir(ana_dir)

    fig_dir = os.path.join(ana_dir, fig_dir_name)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    save_combin_dir = os.path.join(ana_dir, 'combination')  # all test set combined
    if not os.path.exists(save_combin_dir):
        os.mkdir(save_combin_dir)

    save_combin_path = os.path.join(save_combin_dir, '{}.npy')  # store accuracy

    save_sep_dir = os.path.join(ana_dir, 'seperate')  # seperate test set
    if not os.path.exists(save_sep_dir):
        os.mkdir(save_sep_dir)
    save_sep_path = os.path.join(save_sep_dir, '{}.npy')  # store accuracy

    save_extra_dir = os.path.join(ana_dir, 'separate_extra')  # extra energy point
    if not os.path.exists(save_extra_dir):
        os.mkdir(save_extra_dir)
    save_extra_path = os.path.join(save_extra_dir, '{}.npy')  # store accuracy

    # TODO ---------------------------check-----------------------------------------------------------------------------

    signals_dict = {
        2: ['e+', 'pi+'],
        3: ['mu+', 'e+', 'pi+'],
        4: ['mu+', 'e+', 'pi+', 'noise']}
    # combination

    combin_datasets_dir = combin_datasets_dir_dict.get(n_classes)
    combin_datasets_path = os.path.join(combin_datasets_dir, 'imgs.npy')
    combin_labels_path = os.path.join(combin_datasets_dir, 'labels.npy')

    # # extra energy points
    # extra_datasets_dir = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/extra_energy_point'
    # extra_e_pi_proton_datasets_path = os.path.join(extra_datasets_dir, 'ahcal_{}_{}GeV_2cm_1k/datasets.npy')
    # extra_e_pi_proton_labels_path = os.path.join(extra_datasets_dir, 'ahcal_{}_{}GeV_2cm_1k/labels.npy')
    # extra_e_pi_proton_energy_points = sorted([15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 110, 115, 125, 130])
    #
    # extra_mu_datasets_path = os.path.join(extra_datasets_dir, 'ahcal_{}_{}GeV_2cm_1k/datasets.npy')
    # extra_mu_labels_path = os.path.join(extra_datasets_dir, 'ahcal_{}_{}GeV_2cm_1k/labels.npy')
    # extra_mu_energy_points = sorted([100, 120, 130, 140, 150, 170, 180, 190, 200])

    #   distribution
    save_dis_dir = os.path.join(ana_dir, 'ahcal_beam_test_dis')
    if not os.path.exists(save_dis_dir):
        os.mkdir(save_dis_dir)
    save_dis_path = os.path.join(save_dis_dir, '{}_dis.npy')

    dis_datasets_path = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/ahcal_beam_test_mu_e_pi_no_energy/{}/datasets.npy'
    dis_labels_path = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/ahcal_beam_test_mu_e_pi_no_energy/{}/labels.npy'

    # roc

    save_roc_dir = os.path.join(ana_dir, 'roc')
    if not os.path.exists(save_roc_dir):
        os.mkdir(save_roc_dir)
    save_roc_path = os.path.join(save_roc_dir, '{}.npy')
    fpr_path = save_roc_path.format('fpr')
    tpr_path = save_roc_path.format('tpr')
    auroc_path = save_roc_path.format('auroc')

    # TODO -------------------------------------------------------------------------------------------------------------

    ####################################################################################

    # net = net_dict.get(net_used)
    # net_paras = net_para_dict.get(net_used)
    # net = net(**net_paras)

    if torch.cuda.is_available():
        net = net.cuda()
        net.load_state_dict(torch.load(model_path))
        device = 'cuda'
    else:
        device = 'cpu'
        net.load_state_dict(torch.load(model_path, map_location=device))

    signals = signals_dict.get(n_classes)
    #  combination

    if comb_flag:
        # data loader

        loader_test = data_loader_func(combin_datasets_path,
                                       combin_labels_path,
                                       mean_std_static=True,
                                       num_workers=0,
                                       batch_size=128,
                                       max_nodes=max_nodes)

        correct_val = 0.
        total_val = 0.
        with torch.no_grad():
            net.eval()
            for j, (inputs, labels) in enumerate(loader_test):
                # input configuration
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().cpu().numpy()
            acc = "{:.2f}".format(100 * correct_val / total_val)

        np.save(save_combin_path.format('combination'), np.array([acc]))

        predicts = []
        targets = []
        with torch.no_grad():
            net.eval()
            for j, (inputs, labels) in enumerate(loader_test):
                # input configuration
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                # _, predicted = torch.max(outputs.data, 1)
                #
                # for type in range(n_classes):
                #
                #     total_val[type] += (labels[labels==type]).size(0)
                #     correct_val[type] += (predicted[labels==type] == labels[labels==type]).squeeze().sum().cpu().numpy()

                # acc = 100 * correct_val / total_val

                predicts.append(outputs)
                targets.append(labels)
            targets = torch.cat(targets)
            predicts = torch.cat(predicts)

            mca = MulticlassAccuracy(num_classes=n_classes, average=None, threshold=threshold).to(device)
            acc_particles = 100 * mca(predicts, targets).cpu().numpy()
        np.save(save_combin_path.format('acc_particles'), acc_particles)

        save_acc_particle_path = os.path.join(fig_dir, 'acc_particle.png')
        plotACCbar(acc_particles, save_acc_particle_path, threshold)

        # purities = purity_at_thresholds(model=net, dataloader=loader_test, device=device, num_classes=n_classes,
        #                                 thresholds_num=threshold_num)
        # np.save(save_combin_path.format('purities'), purities)
        # for dim, signal in enumerate(signals):
        #     save_purities_path = os.path.join(fig_dir, 'purity_{}.png'.format(signal))
        #     plot_purity_threshold(purities=purities,
        #                           save_path=save_purities_path,
        #                           signal_dict={'name':signal, 'dim':dim},
        #                           threshold_num=threshold_num,
        #                           data_type=data_type)

        # plot roc
        preds = torch.tensor([])
        targets = torch.tensor([])
        with torch.no_grad():
            net.eval()
            for j, (inputs, labels) in enumerate(loader_test):

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
            metric = MulticlassROC(num_classes=n_classes, thresholds=threshold_num, ignore_index=None).to(
                device)
            fprs_, tprs_, thresholds_ = metric(preds, targets)
            fprs = []
            tprs = []
            for i, fpr in enumerate(fprs_):
                fprs.append(fpr.cpu().numpy())
                tprs.append(tprs_[i].cpu().numpy())

            np.array(fprs, dtype=object)
            np.array(tprs, dtype=object)
            np.save(save_roc_path.format('fpr'), fprs)
            np.save(save_roc_path.format('tpr'), tprs)

            mc_auroc = MulticlassAUROC(num_classes=n_classes, average=None, thresholds=None, ignore_index=None)
            auroc = mc_auroc(preds, targets)
            np.save(save_roc_path.format('auroc'), auroc.cpu().numpy())

        save_roc_fig_path = os.path.join(fig_dir, '{}_roc.png')
        save_roc_threshold_path = os.path.join(fig_dir, '{}_threshold.png')
        save_roc_threshold_ratio_path = os.path.join(fig_dir, '{}_ratio_threshold_ann_' + data_type + '.png')

        for dim, signal in enumerate(signals):
            plotROC(fpr_path=fpr_path, tpr_path=tpr_path, auroc_path=auroc_path,
                    signal_dict={'name': signal, 'dim': dim},
                    save_path=save_roc_fig_path.format(signal), data_type=data_type)
            plot_s_b_threshold(fpr_path=fpr_path, tpr_path=tpr_path, signal_dict={'name': signal, 'dim': dim},
                               save_path=save_roc_threshold_path.format(signal), threshold_num=threshold_num,
                               data_type=data_type)
            plot_s_b_ratio_threshold(fpr_path=fpr_path, tpr_path=tpr_path, signal_dict={'name': signal, 'dim': dim},
                                     save_path=save_roc_threshold_ratio_path,
                                     threshold_num=threshold_num,
                                     data_type=data_type)

    if sep_flag:

        # seperate energy points
        sep_datasets_dir = sep_datasets_dir_dict.get(n_classes)

        sep_energy_points = np.linspace(10, 120, 12).astype(np.int64)
        sep_energy_points = np.hstack([np.array([5]), sep_energy_points])
        # caculate seperate acc

        for ep in sep_energy_points:
            # data loader
            sep_img_test_path = os.path.join(sep_datasets_dir, '{}GeV/imgs.npy'.format(ep))
            sep_label_test_path = os.path.join(sep_datasets_dir, '{}GeV/labels.npy'.format(ep))

            loader_test_sep_ = data_loader_func(sep_img_test_path,
                                                sep_label_test_path,
                                                mean_std_static=True,
                                                num_workers=0,
                                                batch_size=128,
                                                max_nodes=max_nodes)

            acc = totalACC(loader_test_sep_, net, device)
            np.save(save_sep_path.format('ep_{}_acc'.format(ep)), np.array([acc]))

            acc_particles_ = ACCParticle(loader_test_sep_, net, device, n_classes, threshold=threshold)
            np.save(save_sep_path.format('ep_{}_acc_particles'.format(ep)), acc_particles_)
            save_acc_particle_path_ = os.path.join(fig_dir, 'acc_particle_ep_{}.png'.format(ep))
            plotACCbar(acc_particles_, save_acc_particle_path_, threshold)

            purities_ = purity_at_thresholds(model=net, dataloader=loader_test_sep_, device=device,
                                             num_classes=n_classes,
                                             thresholds_num=threshold_num)
            np.save(save_sep_path.format('ep_{}_purities'.format(ep)), purities_)
            for signal in signals:
                save_purities_path_ = os.path.join(fig_dir, 'purity_{}_ep_{}.png'.format(signal, ep))
                plot_purity_threshold(purities_, signal, save_purities_path_, threshold_num=threshold_num,
                                      data_type=data_type)

            # plot roc
            save_roc_ep_path = os.path.join(save_roc_dir, 'ep_' + str(ep) + '_{}.npy')
            getROC(loader_test_sep_, net, device, save_roc_ep_path, n_classes, threshold_num=threshold_num)

            save_sep_roc_fig_path = os.path.join(fig_dir, 'ep_' + str(ep) + '_{}_roc.png')
            save_sep_roc_threshold_path = os.path.join(fig_dir, 'ep_' + str(ep) + '_{}_threshold.png')

            for signal in signals:
                plotROC(fpr_path=save_roc_ep_path.format('fpr'), tpr_path=save_roc_ep_path.format('tpr'),
                        auroc_path=save_roc_ep_path.format('auroc'), signal_dict={'name': signal, 'dim': dim},
                        save_path=save_sep_roc_fig_path.format(signal), data_type=data_type)

                plot_s_b_threshold(fpr_path=save_roc_ep_path.format('fpr'), tpr_path=save_roc_ep_path.format('tpr'),
                                   signal_dict={'name': signal, 'dim': dim},
                                   save_path=save_sep_roc_threshold_path.format(signal),
                                   threshold_num=threshold_num, data_type=data_type)
        for signal in signals:
            plot_purity_ep(threshold=threshold,
                           file_lists=[os.path.join(save_sep_dir, 'ep_{}_purities.npy'.format(ep)) for ep in
                                       sep_energy_points],
                           ep_lists=sep_energy_points,
                           signal=signal,
                           save_path=os.path.join(fig_dir, '{}_purity_vs_ep.png'),
                           threshold_num=threshold_num,
                           )

            plot_s_b_ep(threshold=threshold,
                        tpr_file_lists=[os.path.join(save_roc_dir, 'ep_{}_tpr.npy'.format(ep)) for ep in
                                        sep_energy_points],
                        fpr_file_lists=[os.path.join(save_roc_dir, 'ep_{}_fpr.npy'.format(ep)) for ep in
                                        sep_energy_points],
                        ep_lists=sep_energy_points,
                        signal=signal,
                        save_path=os.path.join(fig_dir, '{}_s_b_vs_ep.png'),
                        threshold_num=threshold_num
                        )

    # probability distribution
    if dis_flags:
        particles_dict = signals_dict
        particles = particles_dict.get(n_classes)
        for particle in particles:
            img_dis_path = os.path.join(root_path, dis_datasets_path).format(particle)
            label_dis_path = os.path.join(root_path, dis_labels_path).format(particle)
            loader_dis = data_loader_func(img_dis_path,
                                          label_dis_path,

                                          mean_std_static=True,
                                          num_workers=0,
                                          batch_size=128,
                                          max_nodes=max_nodes)

            pbDisctuibution(loader_dis, net, save_dis_path.format(particle), device)

            # plot probability distribution
            pi_path_dis = save_dis_path.format('pi+')
            e_path_dis = save_dis_path.format('e+')
            mu_path_dis = save_dis_path.format('mu+')
            proton_path_dis = save_dis_path.format('proton')
            save_dis_compare_path = os.path.join(fig_dir, '{}_dis{}{}.png')

            for log in [True, False]:
                for stack in [True, False]:
                    plotDistribution(mu_path=mu_path_dis, e_path=e_path_dis, pi_path=pi_path_dis,
                                     proton_path=proton_path_dis,
                                     log=log, stack=stack, save_path=save_dis_compare_path, n_classes=n_classes)


def read_ann_score(file_pid_path, n_classes=4, rt_df=False):
    branch_list_dict = {
        2: ['ANN_e_plus', 'ANN_pi_plus'],
        3: ['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', ],
        4: ['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise'],
    }
    branch_list = branch_list_dict.get(n_classes)

    ann_pid = ReadRoot(file_path=file_pid_path, tree_name='Calib_Hit', exp=branch_list)

    ann_score = {}
    for branch in branch_list:
        ann_score[branch] = ann_pid.readBranch(branch)

    if rt_df:
        return pd.DataFrame(ann_score)
    else:
        return pd.DataFrame(ann_score).values


def get_ann_info(

        ann_scores,
        raw_labels,
        ann_info_save_dir,
        n_classes,
        ann_signal_label_list,
        effi_points,
        export=True,
        detailed=True
):
    for signal in ann_signal_label_list:

        if signal == 0:
            continue  # TODO temporarily only take 1 as the signal

        ann_threshold_lists = np.sort(ann_scores[:, signal])
        ann_threshold_lists = np.unique(ann_threshold_lists)

        label_include = copy.deepcopy(ann_signal_label_list)
        label_include.remove(signal)

        for b in label_include:
            ann_ana = ANN_ANA(
                ann_scores_path=None,
                ann_scores=ann_scores,
                raw_labels_path=None,
                raw_labels=raw_labels,
                save_dir=ann_info_save_dir,
                ann_threshold_lists=ann_threshold_lists,
                ann_signal_label=signal,
                n_classes=n_classes

            )
            ann_ana.filter_label(label_list=[signal, b])
            ann_ana.export_ann_info(effi_points=effi_points,
                                    export=export,
                                    detailed=detailed)


def ana_info(ana_dir,
             n_classes,
             ann_signal_label_list,
             effi_points,
             export,

             ):
    cols = ['ANN_e', 'ANN_pi']
    ann_scores = pd.read_csv(os.path.join(ana_dir, 'imgs_ANN.csv'))
    raw_labels = ann_scores['particle_label'].values

    get_ann_info(
        ann_scores=ann_scores[cols].values,
        raw_labels=raw_labels,
        ann_info_save_dir=ana_dir,
        n_classes=n_classes,
        ann_signal_label_list=ann_signal_label_list,
        effi_points=effi_points,
        export=export,
        detailed=True
    )



def ann_eval(ckp_dir,
             data_dir_format,
             n_classes,
             net,
             pid_data_loader_func,
             max_nodes,
             ann_signal_label_list,
             effi_points,
             ):
    ann_scores_path_list = []
    cols = ['ANN_e', 'ANN_pi']
    ana_dir = os.path.join(ckp_dir, 'ANA')

    os.makedirs(ana_dir, exist_ok=True)

    for _ in glob.glob(data_dir_format):
        dataset_type = list(_.split('/'))[-1]

        ana_dir = os.path.join(ckp_dir, 'ANA')
        os.makedirs(ana_dir, exist_ok=True)

        pid_tag_dir = os.path.join(ana_dir, 'PIDTags')
        os.makedirs(pid_tag_dir, exist_ok=True)

        pid_tag_dir = os.path.join(pid_tag_dir, dataset_type)
        os.makedirs(pid_tag_dir, exist_ok=True)

        ana_scores_path = os.path.join(pid_tag_dir, 'imgs_ANN.csv')

        ann_scores_path_list.append(ana_scores_path)

        model_path = os.path.join(ckp_dir, get_file_name(ckp_dir))
        npyPID(file_path=os.path.join(_, 'imgs.npy'),
               save_path=ana_scores_path,
               model_path=model_path,
               n_classes=n_classes,
               net=net,
               pid_data_loader_func=pid_data_loader_func,
               df=True,
               labels=np.load(os.path.join(_, 'labels.npy')),
               cols=cols,
               max_nodes=max_nodes,

               )

    df_list_ = []
    for _ in ann_scores_path_list:
        df_ = pd.read_csv(_)
        df_list_.append(df_)

    ann_scores = pd.concat(df_list_)

    ann_scores.to_csv(os.path.join(ana_dir, 'imgs_ANN.csv'), index=False)

    raw_labels = ann_scores['particle_label'].values

    get_ann_info(
        ann_scores=ann_scores[cols].values,
        raw_labels=raw_labels,
        ann_info_save_dir=ana_dir,
        n_classes=n_classes,
        ann_signal_label_list=ann_signal_label_list,
        effi_points=effi_points
    )


if __name__ == '__main__':

 pass