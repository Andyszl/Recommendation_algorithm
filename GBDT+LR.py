‘’‘
本节主要包括两部分
第一为GBDT+LR
第二是分别用GBDT和LR预测
对比结果
’‘’

import pandas as pd
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

#数据读取
train = pd.read_csv("/Users/admin/Desktop/database/diabetes/diabetes_train.txt",header=None,index_col=False)
test = pd.read_csv("/Users/admin/Desktop/database/diabetes/diabetes_test.txt",header=None,index_col=False)
#数据转换
label = train.loc[:,[8]]
data = train.drop(columns=8)

y_test =  test.loc[:,[8]]
X_test =  test.drop(columns=8)

# 将数据保存到LightGBM二进制文件将使加载更快
lgb_train = lgb.Dataset(data, label)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

#定义基本参数进行训练
#建立30颗树，16个叶子结点
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 16,
    'num_trees': 30,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

 
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)
                
                
#得到训练结果--叶子结点
y_pred_train = gbm.predict(data, pred_leaf=True) 
print(np.array(y_pred_train).shape)
#(500, 30)

# 创建 N * num_tress * num_leafs的空矩阵
num_leaf = 16
transformed_training_matrix = np.zeros([len(y_pred_train), len(y_pred_train[0]) * num_leaf],
                                       dtype=np.int64)  
                                       
np.shape(transformed_training_matrix)
#(500, 480)480=30*16

#根据叶子节点位置填入数据
for i in range(0, len(y_pred_train)):
    #找到样本落在的叶子结点，每一个样本由100个节点
    temp = np.arange(len(y_pred_train[0])) * num_leaf + np.array(y_pred_train[i])
    transformed_training_matrix[i][temp] += 1
    
np.shape(transformed_training_matrix)
#(500, 480)
transformed_training_matrix[0].sum()
#30

print('测试数据格式转换')
y_pred_test = gbm.predict(X_test, pred_leaf=True)
 
transformed_testing_matrix = np.zeros([len(y_pred_test), len(y_pred_test[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_pred_test)):
    temp = np.arange(len(y_pred_test[0])) * num_leaf + np.array(y_pred_test[i])
    transformed_testing_matrix[i][temp] += 1
 ##=================================  LR预测 ======================================
lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction
lm.fit(transformed_training_matrix,label)  # fitting the data
y_predict_lr_test = lm.predict(transformed_testing_matrix)   # Give the probabilty on each label
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_predict_lr_test) 
#0.7947761194029851

#在Facebook的paper中，模型使用NE(Normalized Cross-Entropy)，进行评价:
y_pred_lr_test = lm.predict_proba(transformed_testing_matrix)   # 得到预测概率
NE = (-1) / len(y_pred_lr_test) * np.sum(((1+y_test)/2 * np.reshape(np.log(y_pred_lr_test[:,1]),[-1,1]) +  (1-y_test)/2 *np.reshape(np.log(1-y_pred_lr_test[:,1]),[-1,1]))).values[0]
print("Normalized Cross Entropy " + str(NE))
#Normalized Cross Entropy 0.9400278055507182


‘’‘用LR单独预测’‘’
from sklearn.linear_model import LogisticRegression
lm.fit(data,label)  # fitting the data
y_predict = lm.predict(X_test)   
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_predict) 
#0.7126865671641791

‘’‘用GBDT单独预测’‘’
from sklearn.ensemble import GradientBoostingClassifier
gbr = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)
gbr.fit(data, label)
y_gbr1 = gbr.predict(X_test)
acc_test = gbr.score(X_test, y_test)

print(acc_test)
#0.753731343283582









