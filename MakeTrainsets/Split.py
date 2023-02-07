import numpy as np
import os


def splitDatasets(file_path,save_dir,indices_or_sections,shuffle):
    '''
    :param file_path:
    :param save_dir:
    :param indices_or_sections: a list, value in it means the ratio of the split, e.g.[0.1,0.2]=1:1:8
    :param shuffle:
    :return:
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
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

    file_path = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/raw_data/ahcal_e+_100GeV_2cm_10k.npy'
    save_dir = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/trainsets_e+_mu+_pi+/e+/split'

    splitDatasets(file_path=file_path,save_dir=save_dir,indices_or_sections=[0.8,0.9],shuffle=True)


