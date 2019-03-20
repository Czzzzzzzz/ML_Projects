#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:35:25 2018

@author: zhengcao
"""
import numpy as np

class FileHandler:
    def readData(self, fileName, delimiter=","):
        raw_data = open(fileName, "rt")
        data_and_label = np.loadtxt(raw_data, delimiter=delimiter)
        
        return data_and_label