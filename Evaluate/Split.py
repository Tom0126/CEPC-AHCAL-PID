import numpy as np
import os
import argparse
parser = argparse.ArgumentParser()
# base setting
parser.add_argument("--ep", type=str, default=None, help="energy point")
args=parser.parse_args()


def splitDatasets(file_path,save_dir,indices_or_sections,shuffle):
    '''
    :param file_path:
    :param save_dir:
    :param indices_or_sections: a list, value in it means the ratio of the split, e.g.[0.1,0.2]=1:1:8
    :param shuffle:
    :return:
    '''
    data=np.load(file_path)
    if shuffle:
        np.random.shuffle(data)
    num=len(data)
    indices_or_sections=list(map(lambda x: int(x*num),indices_or_sections))
    results=np.split(data,indices_or_sections=indices_or_sections,axis=0)
    for i ,result in enumerate(results):
        save_path=os.path.join(save_dir,'{}.npy'.format(i))
        np.save(save_path,result)

if __name__ == '__main__':
    particle='e+'
    root_dir = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/separate_e+_mu+_pi+'
    file_path = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/raw_data/ahcal_{}_{}GeV_2cm_10k.npy'.format(particle,args.ep)

    save_dir = os.path.join(root_dir,'ahcal_{}_{}GeV_2cm_10k'.format(particle,args.ep))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    splitDatasets(file_path=file_path,save_dir=save_dir,indices_or_sections=[0.8,0.9],shuffle=True)


