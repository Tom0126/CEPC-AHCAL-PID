import numpy as np
import matplotlib.pyplot as plt

def plotROC(fpr_path,tpr_path,auroc_path,bdt_path,signal,save_path, compare=True, bdt_auc=0.995):
    particle_dim={'mu+':0,'e+':1,'pi+':2,'proton':3}
    fprs=np.load(fpr_path, allow_pickle=True)
    tprs=np.load(tpr_path, allow_pickle=True)
    auroc=np.load(auroc_path)

    fpr=fprs[particle_dim.get(signal)]
    tpr = tprs[particle_dim.get(signal)]
    auc=auroc[particle_dim.get(signal)]

    bdt_result=np.loadtxt(bdt_path)
    bdt_result=np.transpose(bdt_result)

    plt.figure(figsize=(6, 5))
    plt.plot(tpr,1-fpr,label='ANN',color='red')

    plt.xlabel('Signal efficiency',fontsize=15)
    plt.ylabel('Background rejection',fontsize=15)

    plt.text(0.1, 0.9, 'CEPC Preliminary', fontsize=15, fontstyle='oblique', fontweight='bold')
    plt.text(0.1, 0.84, 'AHCAL PID', fontsize=12, fontstyle='normal')
    plt.text(0.1, 0.78, '{} Signals'.format(signal), fontsize=12, fontstyle='normal')
    plt.text(0.1, 0.72, 'ANN AUC = {:.3f}'.format(auc), fontsize=12, fontstyle='normal')

    if compare:
        plt.plot(bdt_result[0], bdt_result[1], label='BDT', color='black')
        plt.text(0.1, 0.66, 'BDT AUC = {:.3f}'.format(bdt_auc), fontsize=12, fontstyle='normal')
        compare_tag='_bdt_compare'
    else:
        compare_tag=''
    plt.legend()
    plt.savefig(save_path.format(signal,compare_tag))
    plt.show()

if __name__=='__main__':
    fpr_path='../roc/fpr.npy'
    tpr_path = '../roc/tpr.npy'
    auroc_path = '../roc/auroc.npy'
    bdt_path='../roc/pion_roc_bdt.txt'
    save_path='Fig/ann_bdt_compare.png'
    plotROC(fpr_path=fpr_path,tpr_path=tpr_path,auroc_path=auroc_path,signal='pi+',bdt_path=bdt_path,save_path=save_path)