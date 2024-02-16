#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 19:07
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : ReadRoot.py
# @Software: PyCharm

import uproot


class ReadRoot():

	def __init__(self, file_path, tree_name,start=None,end=None,cut=None, exp=None):
		file = uproot.open(file_path)
		tree = file[tree_name]

		self.tree = tree.arrays(aliases=None, cut=cut, expressions=exp, library="np", entry_start=start,
						 entry_stop=end)

	def readBranch(self, branch):

		return self.tree[branch]



