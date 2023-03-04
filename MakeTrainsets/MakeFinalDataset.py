import numpy as np
import uproot
from ReadRoot import *
import Combine
import Split
import os

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

    labels=np.append(mu_labels,e_labels)
    labels=np.append(labels,pi_labels).astype(np.longlong)

    np.save(save_path,labels)

def makeFinalDatasets(file_path_list,save_dir):
    '''

    :param file_path_list: order should be mu, e+, pion, proton
    :param shuffle:
    :param save_dir:
    :return:
    '''
    datasets = []
    data_length=[]
    for index, file_path in enumerate(file_path_list):
        data = np.load(file_path)
        data_length.append(len(data))
        if index == 0:
            datasets = data
        else:
            datasets = np.append(datasets, data, axis=0)
    datasets_save_path=os.path.join(save_dir,'datasets.npy')
    labels_save_path=os.path.join(save_dir,'labels.npy')
    np.save(datasets_save_path, datasets)
    makeLabels(mu_nums=data_length[0],e_nums=data_length[1],pi_nums=data_length[2],save_path=labels_save_path)


if __name__ == '__main__':
    label_dict = {'mu+': 0,
                  'e+': 1,
                  'pi+': 2,
		  'proton':3}

    energy_points_dict={
        'mu+':[160],#[100, 120, 130, 140, 150, 170, 180, 190, 200],
        'e+':[100, 20, 40, 60, 80, 120, 30, 50, 70, 90],#[15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 110, 115, 125, 130],
        'pi+':[100, 20, 40, 60, 80, 120, 30, 50, 70, 90],#[15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 110, 115, 125, 130],
	'proton':[100, 20, 40, 60, 80, 120, 30, 50, 70, 90],#[15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 110, 115, 125, 130],
    }
    particles=['mu+','e+','pi+']
    dirs = ['train', 'validation', 'test']
    datasets_dir = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/ahcal_beam_test_mu_e_pi_no_energy'


    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)

    for particle in particles:

        # Combine
        #To DO
        load_path = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/raw_data/no_energy/ahcal_{}'.format(
            particle) + '_{}GeV_2cm_no_energy.npy'
        save_dir = os.path.join(datasets_dir, particle)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, 'datasets.npy')
        Combine.combineDatasets(name_lists=energy_points_dict.get(particle), shuffle=False, load_path=load_path,
                                save_path=save_path)
        num=len(np.load(save_path))
        labels=np.ones(num)*label_dict.get(particle)
        np.save(os.path.join(save_dir, 'labels.npy'.format(particle)),labels.astype(np.longlong))

        # Split
        file_path = save_path
        save_dir = os.path.join(save_dir, 'split')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        Split.splitDatasets(file_path=file_path, save_dir=save_dir, indices_or_sections=[0.8, 0.9], shuffle=True)


    # make final datasets
    for i, dir in enumerate(dirs):
        file_path_list=[os.path.join(datasets_dir,'mu+/split/{}.npy'.format(i))
            ,os.path.join(datasets_dir,'e+/split/{}.npy'.format(i))
            ,os.path.join(datasets_dir,'pi+/split/{}.npy'.format(i))]
            # ,os.path.join(datasets_dir,'proton/split/{}.npy'.format(i))]
        save_dir=os.path.join(datasets_dir,dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        makeFinalDatasets(file_path_list,save_dir)
