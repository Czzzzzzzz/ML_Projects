
# coding: utf-8


import numpy as np
import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle


# In[152]:


def cal_ks(label,score):
    fpr,tpr,thresholds= roc_curve(label,score)
    return max(tpr-fpr)
    
def cal_auc_ks(y_true, y_pred, name = None, save=False):
    sample = name + " Sample : %s" % len(y_true)
    auc = name + ' test_set auc : %0.3f' % roc_auc_score(y_true, y_pred)
    ks = name + ' test_set ks  : %0.3f' % cal_ks(y_true,y_pred) 
    print (sample)
    print (auc)
    print (ks)
    print ('----------------cal_auc_ks process successfully!----------------')
    if save:
        if name:
            pass
        else:
            name = ''
        with open(name + '_auc&ks.txt', 'a+') as f:
            f.write(sample + '\n' + auc + '\n' + ks + '\n' + '------------------------------------' + '\n' )
            print ('----------------cal_auc_ks save successfully!----------------')
    return roc_auc_score(y_true, y_pred), cal_ks(y_true,y_pred) 


def read_data(file_name, dims):
    
    with open(file_name, 'r') as f:    
        lines = f.readlines()        
        
        rows, cols = len(lines) - 1, dims + 2
        all_data = np.array([[np.nan for _ in range(cols)] for _ in range(rows)])

        for idx, line in enumerate(lines[1:]):
            data, label = line.split('\t')
            
            all_data[idx, -1] = float(label)
            
            for pair in data.split(' '):
                index = int(pair.split(':')[0])
                val = pair.split(':')[1]
                
                all_data[idx, index] = val
    
    return pd.DataFrame(all_data)

dims = 2839
train = read_data('train.data', dims)
test = read_data('test.data', dims)
feature_violin = pd.read_csv('feature_violin.txt', header = -1)

def feature_select(data, idx):
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    y_name = data.columns[-1]
    
    X = X[idx]
    X[y_name] = y

    return X

# # 使用从lightgbm训练得到的feature重要性来做特征选择, 挑出重要性值大于50的特征

# feature_importance = pd.read_csv('feature_importance_lgb.txt', header=-1)
# feature_importance_threshold = 50

# feature_importance_index = feature_importance[feature_importance.iloc[:, 1] > feature_importance_threshold].iloc[:, 0]

# train = feature_select(train, feature_importance_index)
# test = feature_select(test, feature_importance_index)

# 在使用lightgbm训练得到的feature重要性来做特征选择后, 用violin plot进一步筛选重要特征
train = feature_select(train, feature_violin[0])
test = feature_select(test, feature_violin[0])



# 填充缺失值，对“猜测”的连续变量填充均值，对离散变量填充众数

uniq_counts = np.array([train.iloc[:, c].unique().shape[0] for c in range(train.shape[1])])
categorical_cols = train.columns[uniq_counts < 10]
numerical_cols = train.columns[uniq_counts >= 10]


for cat_col in categorical_cols:
    mode = train[cat_col].mode()[0]
    train[cat_col].fillna(mode, inplace=True)
    test[cat_col].fillna(mode, inplace=True)
    
for num_col in numerical_cols:
    mean = train[num_col].mean()
    train[num_col].fillna(mean, inplace=True)    
    test[num_col].fillna(mean, inplace=True)        


train_X = train.iloc[:, :-1]
train_y = train.iloc[:, -1].astype('int')

test_X = test.iloc[:, :-1]
test_y = test.iloc[:, -1].astype('int')

def restore_model(filename='model.pkl'):
    # and later you can load it
    with open(filename, 'rb') as f:
        clf = pickle.load(f)

    return clf

cls = restore_model('model.pkl')

predictions = cls.predict(test_X)
    
print('=========Results=============')
cal_auc_ks(predictions, test_y, name='results')
# print('confusion matrix:', confusion_matrix(predictions, test_y))
# print('f1 score:', f1_score(predictions, test_y))



