#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 21:30:24 2018

@author: zhengcao
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from DrawingTool import DrawingTool
from FileHandler import FileHandler
from DataHandler import DataHandler

def draw_scatter_matrix(data, label):
    drawingTool = DrawingTool()

    label_name = ['var of WT img', 'var of WT img', 'curtosis of WT img', 'entropy of img']
    fig = drawingTool.scatterplot_matrix_multiclass(np.transpose(data), label, ['r', 'b'], label_name, 
                    marker='+')
    fig.suptitle('Simple Scatterplot Matrix')
    plt.show()
    
def draw_box_plots(data, labels):
    
    data_handler = DataHandler()    
    separated_data = data_handler.separateDataByLabel(data, labels)
    
    label_name = [ "class " + str(int(label)) for label in np.unique(labels)]
    colors = ['lightblue', 'lightgreen']
    variable_name = ['var of WT img', 'var of WT img', 'curtosis of WT img', 'entropy of img']
    
    for i_variables, var_name in zip(range(data.shape[1]), variable_name):
        data_for_i_var = [separated_data[i_labels][:, i_variables] for i_labels in range(np.unique(labels).shape[0])]

        plt.figure(i_variables, figsize=(8, 8))
        plt.ylabel(var_name)
        bplot = plt.boxplot(data_for_i_var, labels=label_name, patch_artist=True)
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

def knn_plot_accuracy(X, y_train_accuracy, y_test_accuracy): 
    
    plt.figure(figsize=(8, 7))
#    plt.plot(ks_inverse, train_accuracy)    
#    plt.plot(ks_inverse, test_accuracy)
    plt.plot(X, y_train_accuracy, label='train error')
    plt.plot(X, y_test_accuracy, label='test error')
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    
    plt.ylabel("error rate")
    plt.xlabel("k")
    
def knn_plot_learn_curve(X, y):
    plt.figure(figsize=(8, 7))
    plt.plot(X, y)
    plt.ylabel('test error')    
    plt.xlabel("N")
    plt.show()
  
def knn_train(train_data, test_data, train_label, test_label):
    """
        Train the model by KNN. Gain the optimal k ranging from the interval [1, train_data.shape[0]]
        and lowest test error
        
        Parameters
        ----------
        train_data
        test_data
        train_label
        test_label
        
        Returns
        -------
        the lowest test error
        
    """
    ks = np.arange(1, train_data.shape[0], 40)
#    ks = np.arange(1, 20, 3)
    train_accuracy = np.array([])
    test_accuracy = np.array([])
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(train_data, train_label)

        train_acc = 1 - accuracy_score(train_label, knn.predict(train_data))        
        test_acc = 1 - accuracy_score(test_label, knn.predict(test_data))
        
        train_accuracy = np.append(train_accuracy, train_acc)
        test_accuracy = np.append(test_accuracy, test_acc)

#        print(train_accuracy)aa
#        print(test_accuracy)
        print("k={0} : train error {1}, test error {2}".format(k, train_acc, test_acc))
    
#    ks_inverse = np.reciprocal(ks, dtype='float')
#    print(ks_inverse)
    
#    knn_plot_accuracy(k, train_accuracy, test_accuracy)
    optimal_k = ks[np.argmin(test_accuracy)]
#    return optimal_k
        
    return np.min(test_accuracy), optimal_k

def knn_compute_confusion(train_data, test_data, train_label, test_label, optimal_k):
    knn = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')
    knn.fit(train_data, train_label)
    
    predicted_test_label = knn.predict(test_data)
    confusion_mat = confusion_matrix(test_label, predicted_test_label)

    tn, fp, fn, tp = confusion_mat.ravel()
    print('Confusion Matrix:\n')
    print(confusion_mat)

    tp_rate = float(tp) / (tp + fn)
    tn_rate = float(tn) / (tn + fp)
    precision = float(tp) / (tp + fp)
    f1 = f1_score(test_label, predicted_test_label)
    print(' true positive rate: {0}\n true negative rate: {1}\n precision: {2}\n f1-score: {3}'.format(tp_rate, tn_rate, precision, f1))

def argmin_last(np_array):
    min_val = np_array[0]
    index = 0
    for i, v in enumerate(np_array):
        if v <= min_val:
            index = i
    
    return index

def knn_train_metrics(train_data, test_data, train_label, test_label):
    metrics = ['manhattan', 'mahalanobis']
    metric_parameters = [None, {'V': np.cov(train_data)}]
    
    ks = np.arange(1, 902, 10)

    manhattan_test_error = np.array([])
#    mahalanobis_test_error = np.array([])
    for k in ks:        
        knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        knn.fit(train_data, train_label)
        
        predicted_label = knn.predict(test_data)
        test_error = 1 - accuracy_score(test_label, predicted_label)        
        manhattan_test_error = np.append(manhattan_test_error, test_error)
    
#        for metric, metric_para in zip(metrics, metric_parameters):
#            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, metric_params=metric_para)
#            knn.fit(train_data, train_label)
#            predicted_label = knn.predict(test_data)
#            test_error = 1 - accuracy_score(test_label, predicted_label)

#            if test_error == metrics[0]:
#                manhattan_test_error = np.append(manhattan_test_error, test_error)
#            else:
#                mahalanobis_test_error = np.append(mahalanobis_test_error, test_error)
    
    plt.figure()
    plt.plot(ks, manhattan_test_error, label='manhattan')
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    
#    plt.plot(ks, mahalanobis_test_error, label='mahalanobis')

    plt.xlabel('K')
    plt.ylabel('test error')
    
    optimal_k = ks[argmin_last(manhattan_test_error)]
    ps = np.power([10]*10, np.arange(0.1, 1.1, 0.1))
    minkowski_test_error = np.array([])
    for p in ps:        
        knn = KNeighborsClassifier(n_neighbors=optimal_k, metric='minkowski', metric_params={'p': p})
        knn.fit(train_data, train_label)
        predicted_label = knn.predict(test_data)
        
        test_error = 1 - accuracy_score(test_label, predicted_label)
        minkowski_test_error = np.append(minkowski_test_error, test_error)
    
    plt.figure()
    plt.plot(ps, minkowski_test_error, label='minkowski')
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    plt.show()
    
    optimal_k = ps[np.argmin(minkowski_test_error)]
    print("optimal k: {0}".format(optimal_k))

def main():
    fileHandler = FileHandler()
    data_and_label = fileHandler.readData("data_banknote_authentication.txt")
    data, labels = data_and_label[:, :-1], np.array(data_and_label[:, -1], dtype=int)
    
#    draw_scatter_matrix(data, label)
    
#    draw_box_plots(data, labels)

    data_handler = DataHandler()    
    train_data, test_data, train_label, test_label =  data_handler.split_train_test(data, labels, 200)
    
    '''
        To draw learning curve
    '''
#    N = np.arange(50, 901, 50)
#    test_errors = np.array([])
#    for n in N:
#        print("N: {0}".format(n / 2))
#        _, train_data_small, _, train_label_small =  data_handler.split_train_test(train_data, train_label, int(n / 2))
#        print(train_data_small.shape, test_data.shape, train_label_small.shape, test_label.shape)
#    
#        lowest_test_error, _ = knn_train(train_data_small, test_data, train_label_small, test_label)
#        test_errors = np.append(test_errors, lowest_test_error)
#    
#    print(test_errors)
#    knn_plot_learn_curve(N, test_errors)
    
#   Given a optimal k and corresponding data, compute the confusion matrix.
#    optimal_k = 1
#    knn_compute_confusion(train_data, test_data, train_label, test_label, optimal_k)
    
    '''
        To try out different metrics
    '''
    print(train_data.shape, train_label.shape)
    knn_train_metrics(train_data, test_data, train_label, test_label)
    
    
    

def test():
    X = np.array([[1., 2, 3],[ 2., 5, 6], [2., 12, 333], [12., 23, 34],[ 2., 55, 6], [2., 12, 333]])
    label = np.array([1, 1, 5, 0, 0, 0])
#    plt.figure()
#    plt.boxplot(a)
#    dic = {label: a[]}
#    data = a[:, 1:]
#    labels = a[:, 0]
#    print(labels.shape)
#    m = DataHandler.separateDataByLabel(data, labels)

#    a = np.random.permutation(6)
#    class_counts = np.bincount(label)
#    print(np.cumsum(class_counts)[0:-1])
#    shuffled_label = label.take(a)
#    print(shuffled_label)
    
#    print(np.random.permutation(label))
    
#    X_train, X_test, y_train, y_test = train_test_split(X, label, stratify=label, test_size=2, shuffle=False)
#    print(X_train, X_test, y_train, y_test)
#    a = []
#    a.extend([[1, 2], [2, 3]])
#    a.extend([[2, 3], [5, 6]])
    
#    print(confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]))
    
#    print(a)
    x = np.arange(0, 5, 1)
    y = np.arange(2, 6, 0.5)
    xx, yy = np.meshgrid(x, y)
    
#    print(np.c_[xx.ravel(), yy.ravel()])
    
#    m = [None, ]
#    a = {'a': 3}
#    print(a)

if __name__ == "__main__":
#    test()
    main()
    
#    print(argmin_last([0, 0, 0, 3]))
    

    
    

    
    