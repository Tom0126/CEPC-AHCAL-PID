#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 20:03
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : variable.py
# @Software: PyCharm

import numpy as np

class ShowerVariable():
    def __init__(self,file_path):

        self.data= np.load(file_path)

    def fired(self):

        layers_fired=[]
        cells_fired=[]
        max_cells_fired=[]

        for event in  self.data:

            hits=event>0

            layers_hit=np.any(hits,axis=(0,1))

            assert layers_hit.shape == (len(layers_hit),)

            layer_fired_num=np.sum(layers_hit!=0)
            layers_fired.append(layer_fired_num)

            cells_fired_num=np.sum(hits!=0)
            cells_fired.append(cells_fired_num)

            max_cells_fired_num=np.amax(np.sum(hits!=0,axis=(0,1)))
            max_cells_fired.append(max_cells_fired_num)

        return np.array(layers_fired), np.array(cells_fired), np.array(max_cells_fired)



if __name__ == '__main__':
    pass
