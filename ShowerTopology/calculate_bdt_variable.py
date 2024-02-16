#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/14 19:55
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : calculate_bdt_variable.py
# @Software: PyCharm
import os
import glob
import numpy as np
import pandas as pd
import uproot
import argparse
import math


# np.set_printoptions(threshold=np.inf)

class ReadRoot():

    def __init__(self, file_path, tree_name, start=None, end=None, cut=None, exp=None):
        file = uproot.open(file_path)
        tree = file[tree_name]

        self.tree = tree.arrays(aliases=None, cut=cut, expressions=exp, library="np", entry_start=start,
                                entry_stop=end)

    def readBranch(self, branch):
        return self.tree[branch]



class BDT_VAR():

    def __init__(self, imgs_path, labels_path):

        self.imgs_path=imgs_path
        self.labels_path=labels_path
        self.bdt_var = dict()
        self.imgs=None
        self.labels=None


        self.center_distance=8.5

        self.rms = lambda x: np.sqrt(np.mean(x * x))
        self.sqr = lambda x: x*x

    def load(self):

        self.imgs=np.load(self.imgs_path)
        if self.labels_path!=None:
            self.labels=np.load(self.labels_path)
            self.bdt_var['Particle_label'] = self.labels
    def to_csv(self, save_dir, file_name='bdt_var.csv'):

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, file_name)

        df = pd.DataFrame(self.bdt_var)
        df.to_csv(save_path, index=False)

    def to_root(self,save_dir, file_name='bdt_var.root', tree_name='Calib_Hit'):

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, file_name)

        file = uproot.recreate(save_path)
        file[tree_name] = self.bdt_var

    def filter_zero_counts(self):
        counts=np.count_nonzero(self.imgs, axis=(1,2,3))
        self.labels[counts==0]=-1

    def filter_noise(self):

        e_dep=np.sum(self.imgs, axis=(1,2,3))

        self.labels[np.logical_and(e_dep<10, e_dep>0)]=3

    def calculate_n_hit(self, grouping_cell_size, img):

        count=0

        for i in range(img.shape[-1]):

            layer=img[:,:,i]
            if np.count_nonzero(layer)>0:

                for j in range(0,img.shape[0]-grouping_cell_size+1, grouping_cell_size):

                    for k in range(img.shape[1] - grouping_cell_size + 1):

                        patch=layer[j:j+grouping_cell_size,k:k+grouping_cell_size]

                        if np.count_nonzero(patch)>0:

                            count+=1
            else:
                continue

        return count


    def get_fd(self, beta:int, alpha:list, fd_name:str):

        fd = list()

        for img in self.imgs:

            fd_one=[]

            img_hit_no=np.count_nonzero(img)
            if img_hit_no==0:
                fd.append(0)
                continue

            if beta == 1:
                n_b= img_hit_no

            else:
                n_b=self.calculate_n_hit(grouping_cell_size=beta, img=img)

            for a in alpha:

                n_a=self.calculate_n_hit(grouping_cell_size=a, img=img)

                fd_one.append(1+math.log(n_b/n_a)/math.log(a))

            fd.append(np.mean(np.array(fd_one)))

        self.bdt_var[fd_name]=np.array(fd)


    def get_e_dep(self):

        self.bdt_var['E_dep']=np.sum(self.imgs,axis=(1,2,3))

    def get_shower_start(self, pad_fired=4):

        shower_starts=42 * np.ones(len(self.imgs))
        hits_count_layer=np.count_nonzero(self.imgs, axis=(1,2))

        for i, counts in enumerate(hits_count_layer):
            for layer in range(38):
                if counts[layer]>pad_fired and counts[layer+1]>pad_fired and counts[layer+2]>pad_fired:
                    shower_starts[i]=layer
                    break

        self.bdt_var['Shower_start']=shower_starts

    def get_shower_end(self, pad_fired=2):

        shower_ends=42 * np.ones(len(self.imgs))
        hits_count_layer=np.count_nonzero(self.imgs, axis=(1,2))

        for i, counts in enumerate(hits_count_layer):
            for layer in range(int(self.bdt_var['Shower_start'][i]),39):
                if counts[layer]<=pad_fired and counts[layer+1]<=pad_fired:
                    shower_ends[i]=layer+1
                    break

        self.bdt_var['Shower_end']=shower_ends


    def get_shower_layer(self,cell_rms=1):

        shower_layer_num = np.zeros(len(self.imgs))

        for i, img in enumerate(self.imgs):
            hits_indexes = np.nonzero(img)

            for layer in np.unique(hits_indexes[2]):

                x_indexes = hits_indexes[0][hits_indexes[2] == layer]
                y_indexes = hits_indexes[1][hits_indexes[2] == layer]

                if self.rms(x_indexes - self.center_distance) > cell_rms and self.rms(
                        y_indexes - self.center_distance) > cell_rms:
                    shower_layer_num[i] += 1

        self.bdt_var['Shower_layer']=shower_layer_num

    def get_hit_layer_no(self):

        hits_count_layer = np.count_nonzero(self.imgs, axis=(1, 2))
        layers_fired = np.sum((hits_count_layer > 0) != 0, axis=1)

        self.bdt_var['layers_fired']=layers_fired

    def get_shower_layer_ratio(self):

        self.bdt_var['Shower_layer_ratio']= np.where(self.bdt_var['layers_fired']!=0,
                                                     self.bdt_var['Shower_layer']/self.bdt_var['layers_fired'], 0 )

    def get_shower_density(self, pad_size=3):

        density=np.zeros(len(self.imgs))

        for i, img in enumerate(self.imgs):
            hits_num=np.count_nonzero(img)

            if hits_num==0:
                continue
            else:
                neib_num = 0
                for layer in np.arange(40)[np.count_nonzero(img, axis=(0,1))>0]:
                    samp_layer=img[:,:,layer]

                    hits_indexes=np.nonzero(samp_layer)

                    samp_layer_pad=np.pad(samp_layer, (int((pad_size-1)/2),int((pad_size-1)/2)), 'wrap')

                    for x_index, y_index in zip(hits_indexes[0], hits_indexes[1]):
                        neib_num+=np.count_nonzero(samp_layer_pad[x_index:x_index+pad_size, y_index:y_index+pad_size])

                density[i]=neib_num/hits_num

        self.bdt_var['Shower_density'] = density

    def get_shower_length(self,pad_fired=4):

        if 'Shower_start' not in self.bdt_var.keys():
            self.get_shower_start(pad_fired=pad_fired)

        shower_starts=self.bdt_var['Shower_start']
        shower_length=np.zeros(len(self.imgs))

        assert len(shower_starts) == len(self.imgs)

        for i, img in enumerate(self.imgs):

            if shower_starts[i] > 40:
                shower_length[i]=42

            else:

                hits_indexes = np.nonzero(img)

                rms_ = []
                shower_layers=np.unique(hits_indexes[2])[np.unique(hits_indexes[2])>shower_starts[i]] # layers after shower start layer

                for layer in shower_layers:

                    x_indexes=hits_indexes[0][hits_indexes[2]==layer]
                    y_indexes=hits_indexes[1][hits_indexes[2]==layer]

                    rms_.append(self.rms(np.sqrt(self.sqr(x_indexes-self.center_distance)+self.sqr(y_indexes-self.center_distance))))

                shower_length[i]= shower_layers[np.argmax(rms_)]-shower_starts[i]

        self.bdt_var['Shower_length'] = shower_length

    def get_hits_no(self):

        self.bdt_var['Hits_no']= np.count_nonzero(self.imgs, axis=(1,2,3))

    def get_shower_radius(self):

        shower_radius=[]

        for i, img in enumerate(self.imgs):

            if np.sum(img)==0:
                shower_radius.append(0)
            else:
                hits_indexes = np.nonzero(img)

                x_indexes=hits_indexes[0]
                y_indexes=hits_indexes[1]

                dist_=np.sqrt(self.sqr(x_indexes-self.center_distance)+self.sqr(y_indexes-self.center_distance))

                shower_radius.append(self.rms(dist_))

        self.bdt_var['Shower_radius'] = np.array(shower_radius)

    def get_e_mean(self):

        self.bdt_var['E_mean']=np.where(self.bdt_var['Hits_no']>0, self.bdt_var['E_dep']/self.bdt_var['Hits_no'], 0)

    def get_x_width(self):

        x_width = []

        for i, img in enumerate(self.imgs):

            if np.sum(img) == 0:
                x_width.append(0)
            else:
                hits_indexes = np.nonzero(img)

                x_indexes = hits_indexes[0]


                x_width.append(self.rms(x_indexes-self.center_distance))

        self.bdt_var['X_width'] = np.array(x_width)

    def get_y_width(self):

        y_width = []

        for i, img in enumerate(self.imgs):

            if np.sum(img) == 0:
                y_width.append(0)
            else:
                hits_indexes = np.nonzero(img)

                y_indexes = hits_indexes[1]


                y_width.append(self.rms(y_indexes-self.center_distance))

        self.bdt_var['Y_width'] = np.array(y_width)

    def get_z_width(self):

        z_width = []

        for i, img in enumerate(self.imgs):

            if np.sum(img) == 0:
                z_width.append(0)
            else:
                hits_indexes = np.nonzero(img)

                z_indexes = hits_indexes[2]


                z_width.append(self.rms(z_indexes))

        self.bdt_var['Z_width'] = np.array(z_width)

class BDT_ROOT(BDT_VAR):

    def __init__(self, imgs_path, labels_path, root_file_dir, ann_file_dir, ann_threshold, e_threshold, gap, tree_name):
        super().__init__(imgs_path, labels_path)

        self.root_file_dir=root_file_dir
        self.ann_file_dir=ann_file_dir
        self.ann_threshold=ann_threshold
        self.e_threshold=e_threshold
        self.gap=gap
        self.tree_name=tree_name


    def load(self,):

        var=[]

        for root_file_path in glob.glob(os.path.join(self.root_file_dir, 'AHCAL*.root')):

            ann_file_path= root_file_path.replace(self.root_file_dir, self.ann_file_dir)
            ann_file_path=ann_file_path.replace('.root', '_ANN.root')

            if os.path.exists(ann_file_path):

                self.bdt_var=dict()
                self.imgs=self.convert_root(root_file_path)
                self.labels=self.get_labels(ann_file_path)

                self.bdt_var['Particle_label'] = self.labels

                self.filter_zero_counts()
                self.filter_noise()
                self.get_shower_density()
                self.get_shower_start()
                self.get_shower_end()
                self.get_shower_layer_ratio()
                self.get_e_dep()
                self.get_shower_length()
                self.get_hits_no()
                self.get_shower_radius()

                df=pd.DataFrame(self.bdt_var)

                var.append(df.copy())

        self.bdt_var=pd.concat(var)




    def convert_root(self, file_path):
        '''
            1: inout: root file
            2: output: numpy array NCHW (,40,18,18)
            Tom's ID: 0:mu+, 1:e+, 2:pi+, 3: noise
            '''

        ahcal = ReadRoot(file_path, self.tree_name, exp=['Hit_X', 'Hit_Y', 'Hit_Z', 'Hit_Energy'])
        x = ahcal.readBranch('Hit_X')
        y = ahcal.readBranch('Hit_Y')
        z = ahcal.readBranch('Hit_Z')
        e = ahcal.readBranch('Hit_Energy')

        # read raw root file

        num_events = len(e)
        assert num_events == len(x)
        assert num_events == len(y)
        assert num_events == len(z)

        # NHWC
        deposits = np.zeros((num_events, 18, 18, 40))

        for i in range(num_events):

            energies_ = e[i]

            x_ = np.around((x[i] + 342.5491) / 40.29964).astype(int)
            y_ = np.around((y[i] + 343.05494) / 40.29964).astype(int)
            z_ = ((z[i]) / self.gap).astype(int)
            num_events_ = len(energies_)
            assert num_events_ == len(x_)
            assert num_events_ == len(y_)
            assert num_events_ == len(z_)

            for j in range(num_events_):
                e_ = energies_[j] if energies_[j] > self.e_threshold else 0
                deposits[i, x_[j], y_[j], z_[j]] += e_

        # NCHW

        return deposits


    def get_labels(self, file_pid_path):
        branch_list = ['ANN_mu_plus', 'ANN_e_plus', 'ANN_pi_plus', 'ANN_noise']
        ann_pid = ReadRoot(file_path=file_pid_path, tree_name='Calib_Hit', exp=branch_list)

        ann_score = {}
        for branch in branch_list:
            ann_score[branch] = ann_pid.readBranch(branch)

        ann_score=pd.DataFrame(ann_score).values

        max_scores, max_labels = np.amax(ann_score, axis=1), np.argmax(ann_score, axis=1)

        threshold_cut = max_scores >= self.ann_threshold

        max_labels[~threshold_cut]=-1

        return max_labels


class BDT_NPY(BDT_ROOT):
    def __init__(self, imgs_path, labels_path, root_file_dir, ann_file_dir, ann_threshold, e_threshold, gap, tree_name):
        super().__init__(imgs_path, labels_path, root_file_dir, ann_file_dir, ann_threshold, e_threshold, gap,
                         tree_name)

    def load(self):

        var = []

        for root_file_path in glob.glob(os.path.join(self.root_file_dir, 'imgs.npy')):

            ann_file_path = root_file_path.replace(self.root_file_dir, self.ann_file_dir)
            ann_file_path = ann_file_path.replace('.npy', '_ANN.root')

            if os.path.exists(ann_file_path):
                self.bdt_var = dict()
                self.imgs = np.load(root_file_path)
                self.labels = self.get_labels(ann_file_path)

                self.bdt_var['Particle_label'] = self.labels

                self.filter_zero_counts()
                self.filter_noise()
                self.get_shower_density()
                self.get_shower_start()
                self.get_shower_layer_ratio()
                self.get_e_dep()
                self.get_shower_length()
                self.get_hits_no()
                self.get_shower_radius()

                df = pd.DataFrame(self.bdt_var)

                var.append(df.copy())


        self.bdt_var = pd.concat(var)








def main(imgs_path, labels_path, save_dir):

    bdt_var=BDT_VAR(imgs_path=imgs_path,
                    labels_path=labels_path,
                    )
    bdt_var.load()

    bdt_var.filter_zero_counts()
    bdt_var.filter_noise()

    bdt_var.get_shower_density()
    bdt_var.get_shower_start()
    bdt_var.get_shower_end()
    bdt_var.get_hit_layer_no()
    bdt_var.get_shower_layer()
    bdt_var.get_shower_layer_ratio()
    bdt_var.get_e_dep()
    bdt_var.get_shower_length()
    bdt_var.get_hits_no()
    bdt_var.get_shower_radius()
    bdt_var.get_e_mean()
    bdt_var.get_x_width()
    bdt_var.get_y_width()
    bdt_var.get_z_width()
    bdt_var.get_fd(beta=1, alpha=[2, 3, 6,9, 18], fd_name='FD_1')
    bdt_var.get_fd(beta=2, alpha=[3, 6, 9, 18], fd_name='FD_2')
    bdt_var.get_fd(beta=3, alpha=[6, 9, 18], fd_name='FD_3')
    bdt_var.get_fd(beta=6, alpha=[9,18], fd_name='FD_6')
    bdt_var.to_csv(save_dir=save_dir)
    # bdt_var.to_root(save_dir=save_dir)

def main_beam(ann_threshold, e_threshold, ep, beam_data_dir, pid_tags_dir, save_dir):



    root_file_dir=os.path.join(beam_data_dir,str(ep)+'GeV')
    ann_file_dir=os.path.join(pid_tags_dir, str(ep)+'GeV')

    bdt_var = BDT_ROOT(imgs_path=None,
                      labels_path=None,
                      root_file_dir=root_file_dir,
                      ann_file_dir=ann_file_dir,
                      ann_threshold=ann_threshold,
                      e_threshold=e_threshold,
                      gap=30,
                      tree_name='Calib_Hit'
                      )
    bdt_var.load()


    bdt_var.to_csv(save_dir=save_dir, file_name='{}GeV.csv'.format(ep))





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument("--ep", type=int, help="energy point")
    parser.add_argument("--string", type=str, help="any string")
    args = parser.parse_args()

    npy_dir=args.string

    save_dir_='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/mc_0720_e_pi_block_1_1'
    os.makedirs(save_dir_,exist_ok=True)

    save_dir=os.path.join(save_dir_, list(npy_dir.split('/'))[-1])

    main(imgs_path=os.path.join(npy_dir, 'imgs.npy'),
         labels_path=os.path.join(npy_dir, 'labels.npy'),
         save_dir=save_dir)


    #
    # eps = np.linspace(10, 120, 12).astype(np.int64)
    # eps = np.hstack([np.array([5, 8]), eps])
    # ep_dir_list=[str(i)+'GeV' for i in eps]
    # save_dir_='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/mc/0515_e_pi'
    # os.makedirs(save_dir_,exist_ok=True)
    # for ep_dir in ep_dir_list:
    #     imgs_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/sep_e_pi/'+ep_dir + '/imgs.npy'
    #     labels_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/sep_e_pi/' + ep_dir + '/labels.npy'
    #     save_dir=os.path.join(save_dir_, ep_dir)
    #
    #     main(imgs_path=imgs_path,
    #          labels_path=labels_path,
    #          save_dir=save_dir
    #          )

    # save_root_dir = '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/ps_2023'
    #
    # if not os.path.exists(save_root_dir):
    #     os.mkdir(save_root_dir)
    #
    # beam_dir_list=[
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/10GeV_pi',
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_e_Run133',
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/5GeV_pi_Run123',
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/8GeV_pi_Run114',
    #     '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/10GeV_mu_Run18'
    #                ]
    # for dir in beam_dir_list:
    #     imgs_path = os.path.join(dir, 'imgs.npy')
    #     labels_path = os.path.join(dir, 'labels.npy')
    #     save_dir = os.path.join(save_root_dir, dir.split('/')[-1])
    #     main(
    #         imgs_path=imgs_path,
    #         labels_path=labels_path,
    #         save_dir=save_dir,
    #     )
    #
    # save_root_dir = '/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023'
    #
    # if not os.path.exists(save_root_dir):
    #     os.mkdir(save_root_dir)
    #
    #
    # beam_dir_list=glob.glob('/hpcfs/cepc/higgsgpu/siyuansong/PID/data/xiaxin_pid/*2023')
    # beam_dir_list.append('/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/100GeV_mu_Run25')
    # for dir in beam_dir_list:
    #     imgs_path=os.path.join(dir, 'imgs.npy')
    #     labels_path=os.path.join(dir, 'labels.npy')
    #     save_dir=os.path.join(save_root_dir, dir.split('/')[-1])
    #     main(
    #         imgs_path=imgs_path,
    #         labels_path=labels_path,
    #         save_dir=save_dir,
    #     )


    # load_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_0720'
    # save_root_dir = os.path.join('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets',
    #                              list(load_root_dir.split('/'))[-1])
    #
    # if not os.path.exists(save_root_dir):
    #     os.mkdir(save_root_dir)
    #
    # for dir in ['Train', 'Validation', 'Test']:
    # # for dir in [ 'TV']:
    #
    #     save_dir=os.path.join(save_root_dir, dir)
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #
    #     load_dir=os.path.join(load_root_dir, dir)
    #
    #
    #     main(
    #         imgs_path=os.path.join(load_dir, 'imgs.npy'),
    #         labels_path=os.path.join(load_dir, 'labels.npy'),
    #         save_dir=save_dir
    #     )
    #
    # load_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/0720_version'
    # save_root_dir=os.path.join('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets',
    #                            list(load_root_dir.split('/'))[-1])
    #
    # if not os.path.exists(save_root_dir):
    #     os.mkdir(save_root_dir)
    #
    # for dir in ['Train', 'Validation', 'Test']:
    # # for dir in ['TV']:
    #
    #     save_dir=os.path.join(save_root_dir, dir)
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #
    #     load_dir=os.path.join(load_root_dir, dir)
    #
    #
    #     main(
    #         imgs_path=os.path.join(load_dir, 'imgs.npy'),
    #         labels_path=os.path.join(load_dir, 'labels.npy'),
    #         save_dir=save_dir
    #     )

    # load_root_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi_2'
    # save_root_dir=os.path.join('/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/datasets',
    #                            list(load_root_dir.split('/'))[-1])
    #
    # if not os.path.exists(save_root_dir):
    #     os.mkdir(save_root_dir)
    #
    # # for dir in ['Train', 'Validation', 'Test']:
    # for dir in ['Test']:
    #
    #     save_dir=os.path.join(save_root_dir, dir)
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #
    #     load_dir=os.path.join(load_root_dir, dir)
    #
    #
    #     main(
    #         imgs_path=os.path.join(load_dir, 'imgs.npy'),
    #         labels_path=os.path.join(load_dir, 'labels.npy'),
    #         save_dir=save_dir
    #     )
    #



    # main_beam(
    #     ann_threshold=0.9475494754947549,
    #     e_threshold=0.2,
    #     ep=args.ep,
    #     beam_data_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V3',
    #     pid_tags_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v3_2022/AHCAL_only/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022/pi_v3_tb'
    # )
    #
    # main_beam(
    #     ann_threshold=0.9211692116921169,
    #     e_threshold=0.2,
    #     ep=args.ep,
    #     beam_data_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V3',
    #     pid_tags_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v3_2022/AHCAL_only/0627_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022/pi_v3_mc'
    # )

    # main_beam(
    #     ann_threshold=0.9211692116921169,
    #     e_threshold=0.2,
    #     ep=args.ep,
    #     beam_data_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL/HCAL_alone/pi+_V3',
    #     pid_tags_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v3_2022/AHCAL_only/0627_res18_epoch_200_lr_1e-06_batch64_optim_SGD_classes_4_ihep_mc_v1',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2022/pi_v3_mc_0627_res18_epoch_200_lr_1e-06_batch64_optim_SGD_classes_4_ihep_mc_v1'
    # )

    # main_beam(
    #     ann_threshold=0.9475494754947549,
    #     e_threshold=0.2,
    #     ep=args.ep,
    #     beam_data_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL_2023/pi-_V4.1',
    #     pid_tags_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v4_2023/AHCAL_only/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_v4_tb'
    # )
    #
    # main_beam(
    #     ann_threshold=0.9211692116921169,
    #     e_threshold=0.2,
    #     ep=args.ep,
    #     beam_data_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/AHCAL_2023/pi-_V4.1',
    #     pid_tags_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/v4_2023/AHCAL_only/0627_res18_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_mc_v1',
    #     save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/beam/sps_2023/pi_v4_mc'
    # )

    # root_file_dir ='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/ckv_fd_ckv_0615/Test'
    # ann_file_dir = '/hpcfs/cepc/higgsgpu/siyuansong/PID/PIDTags/ckv_0615/Test/0615_res_epoch_200_lr_0.0001_batch32_optim_SGD_classes_4_ihep_v1'
    #
    # bdt_var = BDT_NPY(imgs_path=None,
    #                    labels_path=None,
    #                    root_file_dir=root_file_dir,
    #                    ann_file_dir=ann_file_dir,
    #                    ann_threshold=0.9475494754947549,
    #                    e_threshold=None,
    #                    gap=None,
    #                    tree_name='Calib_Hit'
    #                    )
    # bdt_var.load()
    #
    # bdt_var.to_csv(save_dir='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/Result/temp/tb_test')

    pass
