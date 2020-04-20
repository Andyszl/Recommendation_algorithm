import xgboost
print(xgboost.__version__)
#'0.80'

#导入相关包
from  sklearn import datasets 
import pandas as pd 
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#导入数据
train = pd.read_csv("/Users/admin/Desktop/database/diabetes/diabetes_train.txt",header=None,index_col=False)
test = pd.read_csv("/Users/admin/Desktop/database/diabetes/diabetes_test.txt",header=None,index_col=False)

#数据转换
label = train.loc[:,[8]].values.reshape(-1,1)
data = train.drop(columns=8).values.reshape(-1,8)

y_test =  test.loc[:,[8]].values.reshape(-1,1)
X_test =  test.drop(columns=8).values.reshape(-1,8)

#dmatrix 格式 在xgboost当中运行速度更快，性能更好。
dtrain = xgb.DMatrix(data,label)
dtest = xgb.DMatrix(X_test,y_test)

‘’‘
Xgboost参数

'booster':'gbtree',
'objective': 'multi:softmax', 多分类的问题
'num_class':10, 类别数，与 multisoftmax 并用
'gamma':损失下降多少才进行分裂
'max_depth':12, 构建树的深度，越大越容易过拟合
'lambda':2, 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, 随机采样训练样本
'colsample_bytree':0.7, 生成树时进行的列采样
'min_child_weight':3, 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束
'silent':0 ,设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, 如同学习率
'seed':1000,
'nthread':7, cpu 线程数
’‘’

xgb_params = {
    'seed': 0,
    'eta': 0.1,
    'colsample_bytree': 0.5,
    'silent': 1,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'max_depth': 5,
    'min_child_weight': 3
}

#交叉验证
bst_cv1 = xgb.cv(xgb_params, dtrain, num_boost_round=50, nfold=3, seed=0, 
                 maximize=False, early_stopping_rounds=10)


plt.figure()
bst_cv1[['train-rmse-mean', 'test-rmse-mean']].plot()


#训练
bst = xgb.train(xgb_params, dtrain, 100, evals=[(dtrain,'train'), (dtest,'test')])

from sklearn import metrics
preds = bst.predict(dtest)
auc = metrics.roc_auc_score(y_test, preds)

#auc= 0.8334398159979555
