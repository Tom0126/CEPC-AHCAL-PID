import numpy as np

def combineDatasets(name_lists,shuffle,load_path,save_path):
    '''
    load_path:xxx/xx{}.npy
    '''
    datasets=[]

    for index,name in enumerate(name_lists):
        file_name=load_path.format(name)
        data=np.load(file_name)
        if index==0:
            datasets=data
        else:
            datasets=np.append(datasets,data,axis=0)
    if shuffle:
        np.random.shuffle(datasets)
    np.save(save_path,datasets)


if __name__ == '__main__':
    load_path = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/raw_data/ahcal_e+_{}GeV_2cm_10k.npy'
    save_path = '/lustre/collider/songsiyuan/CEPC/PID/Trainsets/trainsets_e+_mu+_pi+/e/e+_all.npy'
    combineDatasets(name_lists=[100, 20, 40, 60, 80, 120, 30, 50, 70, 90],shuffle=False,load_path=load_path,save_path=save_path)
