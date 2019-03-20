#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:00:05 2018

@author: zhengcao
"""

import numpy as np

class DataHandler:
    def separateDataByLabel(self, data, labels):
        
#        dic = {label: data[labels == label] for label in np.unique(labels)}
        separatedData = [data[labels == label] for label in np.unique(labels)]
        
        return separatedData
    
    def split_train_test(self, data, labels, n_test):
        '''
            In this method, we assume that the labels is sorted by ascending order
            And we split first n_test data in each class as our test data
            
            Parameters
            ----------
            data: array[n, m]
                all of data
            labels: array[m]
                labels of overall data
            n_test: int
                first n_test rows in each class will be merged into test data.
                
            Return Values
            -------------
            train_data
            test_data
            train_labels
            test_labels
            
        '''
        separated_data = self.separateDataByLabel(data, labels)
        separated_labels = self.separateDataByLabel(labels, labels)

        train_data = []
        test_data = []
        train_labels = []
        test_labels = []        
        
        for class_data, class_label in zip(separated_data, separated_labels):
            train_data.extend(class_data[n_test:, :])
            test_data.extend(class_data[:n_test, :])
            train_labels.extend(class_label[n_test:])
            test_labels.extend(class_label[: n_test])
        
        return np.array(train_data), np.array(test_data), np.array(train_labels), np.array(test_labels)
        
            
        
            