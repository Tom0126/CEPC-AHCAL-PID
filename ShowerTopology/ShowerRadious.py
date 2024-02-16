#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/12 22:21
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : ShowerRadious.py
# @Software: PyCharm
import numpy as np
import uproot
import matplotlib.pyplot as plt


class ReadRoot():
    def __init__(self,file_path, tree_name):
        file=uproot.open(file_path)
        tree=file[tree_name]
        self.tree=tree.arrays(library="np")

    def readBranch(self,branch):
        return self.tree[branch]

def plotShowerRadius(file_path):
    data=np.load(file_path)
    num=len(data)
    radius=[]
    for event in data:
        tags=np.where(event!=0)
        x_index = tags[0]-8.5
        y_index = tags[1]-8.5
        z_index = tags[2]

        r=np.sqrt(x_index**2+y_index**2)
        radius.append(np.mean(r))
    scale=40.29964
    plt.hist(radius*scale,bins=100,range=[0,6.25])
    plt.savefig('radius.png')
    plt.show()

if __name__ == '__main__':
    file_path='/cefs/higgs/siyuansong/PID/PIDTags/npy/run88.npy'
    plotShowerRadius(file_path)
    pass
