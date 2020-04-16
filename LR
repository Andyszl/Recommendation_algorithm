#tensorflow逻辑回归实现
import tensorflow as tf
import  numpy as np
from tensorflow.python.framework import ops 
from sklearn.preprocessing import  OneHotEncoder
import pandas as pd
ops.reset_default_graph()#清空之前缓存

‘’‘参数设定’‘’
#特征数量
n_features = 8
#label个数
n_class = 2

#定义训练轮数
training_steps = 1000
#学习率
learning_rate=0.01

#导入数据
train = pd.read_csv("/Users/admin/Desktop/database/diabetes/diabetes_train.txt",header=None,index_col=False)
test = pd.read_csv("/Users/admin/Desktop/database/diabetes/diabetes_test.txt",header=None,index_col=False)
#数据转换

label = train.loc[:,[8]].values.reshape(-1,1)
data = train.drop(columns=8).values.reshape(-1,n_features)

y_test =  test.loc[:,[8]].values.reshape(-1,1)
X_test =  test.drop(columns=8).values.reshape(-1,n_features)

#one-hot编码
enc = OneHotEncoder()
#训练
enc.fit(label)
enc.fit(y_test)
#转换成array
label=enc.transform(label).toarray() 
y_test =enc.transform(y_test).toarray() 

ops.reset_default_graph()
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_class])

# 模型参数
W = tf.Variable(tf.zeros([n_features, n_class]))
b = tf.Variable(tf.zeros([n_class]))
# W = tf.Variable(tf.truncated_normal([n_features, n_class-1]))
# b = tf.Variable(tf.truncated_normal([n_class]))

# 定义模型，此处使用与线性回归一样的定义
# 因为在后面定义损失的时候会加上映射
pred = tf.matmul(x, W) + b
print("pred",np.shape(pred))
# 定义损失函数
error_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.argmax(y,1)))
tf.add_to_collection("losses", error_loss)      #加入集合的操作

#在权重参数上实现L2正则化
regularizer = tf.contrib.layers.l2_regularizer(0.1)
regularization = regularizer(W)
tf.add_to_collection("losses",regularization)     #加入集合的操作

#get_collection()函数获取指定集合中的所有个体，这里是获取所有损失值
#并在add_n()函数中进行加和运算
loss = tf.add_n(tf.get_collection("losses"))

#定义一个优化器，学习率为固定为0.01，注意在实际应用中这个学习率数值应该大于0.01
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 准确率
pred=tf.nn.sigmoid(pred,name='score')
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #在for循环内进行30000训练
    for i in range(training_steps):
        
        sess.run(train_op, feed_dict={x: data, y: label})

        #训练30000轮，但每隔2000轮就输出一次loss的值
        if i % 100 == 0:
            loss_value,auc = sess.run([loss,accuracy], feed_dict={x: data, y: label})
            print("After %d steps, loss_value is: %f" % (i,loss_value))
            print("After %d steps, accuracy is: %f" % (i,auc))
    print(np.shape(X_test)) 
    print(np.shape(y_test)) 
    print("Testing Accuracy:", accuracy.eval({x: X_test, y:y_test}))



out：
pred (?, 2)
After 0 steps, loss_value is: 3.006452
After 0 steps, accuracy is: 0.636000
After 100 steps, loss_value is: 0.600458
After 100 steps, accuracy is: 0.704000
After 200 steps, loss_value is: 0.583874
After 200 steps, accuracy is: 0.718000
After 300 steps, loss_value is: 0.566768
After 300 steps, accuracy is: 0.744000
After 400 steps, loss_value is: 0.551365
After 400 steps, accuracy is: 0.740000
After 500 steps, loss_value is: 0.538515
After 500 steps, accuracy is: 0.748000
After 600 steps, loss_value is: 0.528287
After 600 steps, accuracy is: 0.746000
After 700 steps, loss_value is: 0.520387
After 700 steps, accuracy is: 0.750000
After 800 steps, loss_value is: 0.514410
After 800 steps, accuracy is: 0.760000
After 900 steps, loss_value is: 0.513852
After 900 steps, accuracy is: 0.762000
(268, 8)
(268, 2)
Testing Accuracy: 0.80597013
