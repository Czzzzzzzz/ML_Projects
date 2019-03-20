#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:01:44 2018

@author: zhengcao
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools

class DrawingTool:
    def scatterplot_matrix(self, data, names, **kwargs):
        """
        Parameters
        -----------
        data: array, shape[variance, size of samples]
            the data needed to be ploted in the figure.
        names: array, shape[variance]
            the name of variables
        **kwargs:
            attributes for setting the style of scatter plot matrix
        
        Examples input
        ------------
            fig = drawingTool.scatterplot_matrix(data, ['mpg', 'disp', 'drat', 'wt'], 
                    linestyle='none', marker='o', color='black', mfc='none')
        """
        numvars, numdata = data.shape
        fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(16,16))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
        for ax in axes.flat:
            # Hide all ticks and labels
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    
            # Set up ticks only on one side for the "edge" subplots...
            if ax.is_first_col():
                ax.yaxis.set_ticks_position('left')
            if ax.is_last_col():
                ax.yaxis.set_ticks_position('right')
            if ax.is_first_row():
                ax.xaxis.set_ticks_position('top')
            if ax.is_last_row():
                ax.xaxis.set_ticks_position('bottom')
    
        # Plot the data.
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i,j), (j,i)]:
                axes[x,y].plot(data[x], data[y], **kwargs)
    
        # Label the diagonal subplots...
        for i, label in enumerate(names):
            axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center')
    
        # Turn on the proper x or y axes ticks.
        for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
            axes[j,i].xaxis.set_visible(True)
            axes[i,j].yaxis.set_visible(True)
    
        return fig  


    def scatterplot_matrix_multiclass(self, data, labels, colors, names, **kwargs):
        """
        Sample input:
            fig = drawingTool.scatterplot_matrix_multiclass(data, label, ['r', 'b'], ['mpg', 'disp', 'drat', 'wt'], 
                    marker='o')
        """
        numvars, numdata = data.shape
        fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(16,16))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
        for ax in axes.flat:
            # Hide all ticks and labels
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    
            # Set up ticks only on one side for the "edge" subplots...
            if ax.is_first_col():
                ax.yaxis.set_ticks_position('left')
            if ax.is_last_col():
                ax.yaxis.set_ticks_position('right')
            if ax.is_first_row():
                ax.xaxis.set_ticks_position('top')
            if ax.is_last_row():
                ax.xaxis.set_ticks_position('bottom')
    
        # Plot the data.
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i,j), (j,i)]:
                for i, (row_x_colval, row_y_colval) in enumerate(zip(data[x], data[y])):
                    axes[x,y].scatter(row_x_colval, row_y_colval, color=colors[int(labels[i])], **kwargs)
    
        # Label the diagonal subplots...
        for i, label in enumerate(names):
            axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center')
    
        # Turn on the proper x or y axes ticks.
        for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
            axes[j,i].xaxis.set_visible(True)
            axes[i,j].yaxis.set_visible(True)
    
        return fig         