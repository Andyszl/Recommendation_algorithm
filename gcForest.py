from gcforest.gcforest import GCForest
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
train = pd.read_csv("/Users/admin/Desktop/database/diabetes/diabetes_train.txt",header=None,index_col=False)
test = pd.read_csv("/Users/admin/Desktop/database/diabetes/diabetes_test.txt",header=None,index_col=False)
#数据转换


label = train.loc[:,[8]].values.reshape(-1)
data = train.drop(columns=8).values.reshape(-1,8)

y_test =  test.loc[:,[8]].values.reshape(-1)
X_test =  test.drop(columns=8).values.reshape(-1,8)


‘’‘
max_depth: 决策树最大深度。默认为"None"，决策树在建立子树的时候不会限制子树的深度这样建树时，会使每一个叶节点只有一个类别，或是达到min_samples_split。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。
estimators表示选择的分类器
n_estimators 为森林里的树的数量
n_jobs: int (default=1)随机森林训练和预测的并行数量，如果等于-1，则作业的数量设置为核心的数量。
’‘’

#训练的配置，采用默认的模型-即原库代码实现方式
def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0  # 0 or 1
    ca_config["max_layers"] = 100 # 最大的层数，layer对应论文中的level
    ca_config["early_stopping_rounds"] = 3 #如果出现某层的三层以内的准确率都没有提升，层中止
    ca_config["n_classes"] = 2 #判别的类别数量
    ca_config["estimators"] = []
    
    ca_config["estimators"].append({"n_folds": 2, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 2, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 2, "type": "LogisticRegression"})
    config["cascade"] = ca_config #共使用了3个基学习器
    return config
    
config=get_toy_config()
gc = GCForest(config)
#X_train_enc是每个模型最后一层输出的结果，每一个类别的可能性
X_train_enc = gc.fit_transform(data, label)
#会打印训练信息，只复制最后两行信息查看
‘’‘
[ 2020-04-21 17:48:27,723][cascade_classifier.calc_accuracy] Accuracy(layer_3 - train.classifier_average)=72.00%
[ 2020-04-21 17:48:27,725][cascade_classifier.fit_transform] [Result][Optimal Level Detected] opt_layer_num=1, accuracy_train=73.80%, accuracy_test=0.00%
’‘’
# 模型预测
y_pred = gc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
#Test Accuracy of GcForest = 82.84 %

#可以使用gcForest得到的X_enc数据进行其他模型的训练比如xgboost/RF
#数据链接
X_test_enc = gc.transform(X_test)
X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
X_train_origin = data.reshape((data.shape[0], -1))
X_test_origin = X_test.reshape((X_test.shape[0], -1))
X_train_enc = np.hstack((X_train_origin, X_train_enc))
X_test_enc = np.hstack((X_test_origin, X_test_enc))

print("X_train_enc.shape={}, X_test_enc.shape={}".format(X_train_enc.shape,X_test_enc.shape))
#X_train_enc.shape=(500, 14), X_test_enc.shape=(268, 14)

# 训练一个RF
clf = RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1)
clf.fit(X_train_enc, label)
y_pred = clf.predict(X_test_enc)
acc1 = accuracy_score(y_test, y_pred)
print("Test Accuracy of Other classifier using gcforest's X_encode = {:.2f} %".format(acc1 * 100))
#Test Accuracy of Other classifier using gcforest's X_encode = 79.48 %

#保存模型
mport pickle
with open("test.pkl", "wb") as f:
    pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)

#读取模型
with open("test.pkl", "rb") as f:
    gc = pickle.load(f)
y_pred = gc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy of GcForest (save and load) = {:.2f} %".format(acc * 100))
#Test Accuracy of GcForest (save and load) = 82.84 %




