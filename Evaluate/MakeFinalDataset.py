import numpy as np
import uproot
import Combine
import Split
from ReadRoot import *
import os

import argparse
parser = argparse.ArgumentParser()
# base setting
parser.add_argument("--ep", type=str, default=None, help="energy point")
parser.add_argument("--p_type", type=str, help="particle type")
args=parser.parse_args()

def makeLabels(mu_nums,e_nums,pi_nums,save_path):
    '''

    :param mu_nums:
    :param e_nums:
    :param pi_nums:
    :param save_path:
    :return:
    mu+:0
    e++:1
    pi+:2
    proton:3
    '''
    mu_labels=np.zeros(mu_nums)
    e_labels=np.ones(e_nums)
    pi_labels=np.ones(pi_nums)*2
    # proton_labels=np.ones(proton_nums)*3
    labels=np.append(mu_labels,e_labels)
    labels=np.append(labels,pi_labels).astype(np.longlong)
    np.save(save_path,labels)

def makeFinalDatasets(file_path,label,save_dir,save_name):
    '''

    param file_path_list: order should be mu, e+, pion
    :param shuffle:
    :param save_dir:
    :return:
    '''

    data = np.load(file_path)
    num=len(data)
    datasets_save_path=os.path.join(save_dir,save_name.format('datasets'))
    labels_save_path=os.path.join(save_dir,save_name.format('labels'))
    np.save(datasets_save_path, data)
    labels=np.ones(num)*label
    labels=labels.astype(np.longlong)
    np.save(labels_save_path,labels)



if __name__ == '__main__':
    label_dict={'mu+':0,
                'e+':1,
                'pi+':2,
                'proton':3}

    particle = 'mu+'


    file_path='/lustre/collider/songsiyuan/CEPC/PID/Trainsets/raw_data/no_energy/ahcal_{}_{}GeV_2cm_no_energy.npy'.format(particle,args.ep)
    save_dir='/lustre/collider/songsiyuan/CEPC/PID/Trainsets/ahcal_beam_test_mu_e_pi_no_energy_seperate/ahcal_{}_{}GeV_2cm_no_energy'.format(particle,args.ep)

    label=label_dict.get(particle)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_name='{}.npy'

    makeFinalDatasets(file_path=file_path,label=label,save_dir=save_dir,save_name=save_name)

