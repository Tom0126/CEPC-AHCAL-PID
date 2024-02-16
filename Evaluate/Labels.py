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

if __name__ == '__main__':
    file_path = '/Users/songsiyuan/PID/File/e+/hcal_e+_80GeV_2cm_10k.root'
    save_path = 'File/e_test.npy'