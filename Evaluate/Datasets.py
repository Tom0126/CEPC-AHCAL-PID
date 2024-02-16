import numpy as np
def makeLabels(mu_nums,e_nums,pi_nums,save_path):
    '''

    :param mu_nums:
    :param e_nums:
    :param pi_nums:
    :param save_path:
    :return:
    mu+:0
    e+:1
    pi+:2
    '''
    mu_labels=np.zeros(mu_nums)
    e_labels=np.ones(e_nums)
    pi_labels=np.ones(pi_nums)*2
    labels=np.append(mu_labels,e_labels)
    labels=np.append(labels,pi_labels).astype(np.longlong)
    np.save(save_path,labels)

def combineDatasets(num,shuffle,save_path):
    datasets=[]
    for i in range(num):
        data=np.load(num)
        datasets=np.append(datasets,data,axis=0)
    if shuffle:
        np.random.shuffle(datasets)
    np.save(save_path,datasets)

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
    file_path = '/Users/songsiyuan/PID/File/e+/hcal_e+_80GeV_2cm_10k.root'
    save_path = 'File/e_test.npy'
    # makeDatasets(file_path, save_path)
    save_label_path = 'File/e_label_test.npy'
    makeLabels(3000,4000,3000,save_label_path)

    splitDatasets(save_path,'File',[0.1,0.2],shuffle=True)
    file0=np.load('File/0.npy')
    file1 = np.load('File/1.npy')
    file2 = np.load('File/2.npy')
    print(file2.shape)
