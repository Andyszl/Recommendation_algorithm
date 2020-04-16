import tensorflow as tf
import  numpy as np
import pandas as pd
from tensorflow.python.framework import ops 
ops.reset_default_graph()

from sklearn.preprocessing import  OneHotEncoder
import pandas as pd
#参数设定
#特征数量
n_features = 8
#label个数
n_class = 2

#定义训练轮数
training_steps = 1000
#学习率
learning_rate=0.01

#隐层K
fv=20

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
with tf.name_scope("Input"):
    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_class])


# 模型参数parameter
with tf.name_scope("Parameter"):
    W = tf.Variable(tf.zeros([n_features, n_class]),name="w")
    b = tf.Variable(tf.zeros([n_class]),name="b")
    v = tf.Variable(tf.zeros([n_features, fv]),name="V")
    # W = tf.Variable(tf.truncated_normal([n_features, n_class-1]))
    # b = tf.Variable(tf.truncated_normal([n_class]))

    # 定义模型，此处使用与线性回归一样的定义
    # 因为在后面定义损失的时候会加上映射
with tf.name_scope("Prediction"):
    
    Y_liner = tf.matmul(x, W) + b
    #0.5*((sum(v*x))^2 - sum((v*x)^2)) 
    Y_pair = 0.5 * tf.reduce_sum(
        tf.subtract(tf.pow(tf.matmul(x, v), 2),#(sum(v*x))^2
                tf.matmul(tf.pow(x, 2),tf.pow(v, 2))),#sum((v*x)^2)
        axis=1,
        keep_dims=True)
    
    pred= tf.add(Y_liner, Y_pair)
    

# 定义损失函数
with tf.name_scope("losses"):
    with tf.name_scope("error_loss"):
        print("pred",tf.shape(pred))
        print("y",tf.shape(y))
        error_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(pred, [-1]), labels=tf.reshape(y, [-1])))
    tf.add_to_collection("losses", error_loss)      #加入集合的操作

    #在权重参数上实现L2正则化
    with tf.name_scope("regularization"):
        regularizer = tf.contrib.layers.l2_regularizer(0.01)
        regularization = regularizer(W)+regularizer(v)
    tf.add_to_collection("losses",regularization)     #加入集合的操作

    #get_collection()函数获取指定集合中的所有个体，这里是获取所有损失值
    #并在add_n()函数中进行加和运算
    loss = tf.add_n(tf.get_collection("losses"))

#定义一个优化器，学习率为固定为0.01，注意在实际应用中这个学习率数值应该大于0.01
with tf.name_scope("Train"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 准确率
with tf.name_scope("accuracy"):

    correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.histogram("accuracy",accuracy)
    tf.summary.scalar("accuracy",accuracy)

merged=tf.summary.merge_all()

with tf.Session() as sess:

    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter("./log",sess.graph)
    #在for循环内进行30000训练
    for i in range(training_steps):
        sess.run(train_op, feed_dict={x: data, y: label})
        
        loss_value = sess.run(loss, feed_dict={x: data, y: label})
        summary,voliadata_accuracy=sess.run([merged,accuracy],feed_dict={x: data, y: label})
        writer.add_summary(summary,i)

        #训练30000轮，但每隔2000轮就输出一次loss的值
        if i % 100 == 0 or i <= 100:
            loss_value = sess.run(loss, feed_dict={x: data, y: label})
            
            print("After %d steps, loss_value is: %f" % (i,loss_value))
            print("After %d trainging steps ,validation accuarcy is %g%%"%(i,voliadata_accuracy*100))
        #xs,ys =data.train.next_batch(200)
        #sess.run(train_op,feed_dict={x:xs,y:ys})
  
    print("Testing Accuracyis %g%%"%(accuracy.eval({x: X_test, y:y_test})*100))
writer.close()

out:
pred Tensor("losses/error_loss/Shape:0", shape=(2,), dtype=int32)
y Tensor("losses/error_loss/Shape_1:0", shape=(2,), dtype=int32)
After 0 steps, loss_value is: 1.548657
After 0 trainging steps ,validation accuarcy is 63.6%
After 1 steps, loss_value is: 0.843941
After 1 trainging steps ,validation accuarcy is 63.6%
After 2 steps, loss_value is: 0.920532
After 2 trainging steps ,validation accuarcy is 36.4%
After 3 steps, loss_value is: 1.096752
After 3 trainging steps ,validation accuarcy is 36.4%
After 4 steps, loss_value is: 0.820558
After 4 trainging steps ,validation accuarcy is 37.2%
After 5 steps, loss_value is: 0.659492
After 5 trainging steps ,validation accuarcy is 64.2%
After 6 steps, loss_value is: 0.827998
After 6 trainging steps ,validation accuarcy is 63.6%
After 7 steps, loss_value is: 0.901290
After 7 trainging steps ,validation accuarcy is 63.6%
After 8 steps, loss_value is: 0.793194
After 8 trainging steps ,validation accuarcy is 63.6%
After 9 steps, loss_value is: 0.659128
After 9 trainging steps ,validation accuarcy is 63%
After 10 steps, loss_value is: 0.688714
After 10 trainging steps ,validation accuarcy is 62.2%
After 11 steps, loss_value is: 0.779658
After 11 trainging steps ,validation accuarcy is 50.8%
After 12 steps, loss_value is: 0.767410
After 12 trainging steps ,validation accuarcy is 51%
After 13 steps, loss_value is: 0.673346
After 13 trainging steps ,validation accuarcy is 63.4%
After 14 steps, loss_value is: 0.632302
After 14 trainging steps ,validation accuarcy is 64.6%
After 15 steps, loss_value is: 0.687855
After 15 trainging steps ,validation accuarcy is 63.8%
After 16 steps, loss_value is: 0.723137
After 16 trainging steps ,validation accuarcy is 63.8%
After 17 steps, loss_value is: 0.686575
After 17 trainging steps ,validation accuarcy is 63.6%
After 18 steps, loss_value is: 0.631938
After 18 trainging steps ,validation accuarcy is 65%
After 19 steps, loss_value is: 0.638120
After 19 trainging steps ,validation accuarcy is 67.6%
After 20 steps, loss_value is: 0.678513
After 20 trainging steps ,validation accuarcy is 61%
After 21 steps, loss_value is: 0.676416
After 21 trainging steps ,validation accuarcy is 61.6%
After 22 steps, loss_value is: 0.637045
After 22 trainging steps ,validation accuarcy is 67.2%
After 23 steps, loss_value is: 0.622274
After 23 trainging steps ,validation accuarcy is 67.6%
After 24 steps, loss_value is: 0.645776
After 24 trainging steps ,validation accuarcy is 64%
After 25 steps, loss_value is: 0.658433
After 25 trainging steps ,validation accuarcy is 63.4%
After 26 steps, loss_value is: 0.639243
After 26 trainging steps ,validation accuarcy is 64.6%
After 27 steps, loss_value is: 0.619695
After 27 trainging steps ,validation accuarcy is 67.4%
After 28 steps, loss_value is: 0.627837
After 28 trainging steps ,validation accuarcy is 67.4%
After 29 steps, loss_value is: 0.642433
After 29 trainging steps ,validation accuarcy is 67%
After 30 steps, loss_value is: 0.636208
After 30 trainging steps ,validation accuarcy is 67.6%
After 31 steps, loss_value is: 0.619665
After 31 trainging steps ,validation accuarcy is 66.8%
After 32 steps, loss_value is: 0.618746
After 32 trainging steps ,validation accuarcy is 67.8%
After 33 steps, loss_value is: 0.629764
After 33 trainging steps ,validation accuarcy is 66.2%
After 34 steps, loss_value is: 0.629564
After 34 trainging steps ,validation accuarcy is 66.2%
After 35 steps, loss_value is: 0.618125
After 35 trainging steps ,validation accuarcy is 68.6%
After 36 steps, loss_value is: 0.615013
After 36 trainging steps ,validation accuarcy is 69.2%
After 37 steps, loss_value is: 0.622566
After 37 trainging steps ,validation accuarcy is 68.2%
After 38 steps, loss_value is: 0.623844
After 38 trainging steps ,validation accuarcy is 67.2%
After 39 steps, loss_value is: 0.616053
After 39 trainging steps ,validation accuarcy is 68.4%
After 40 steps, loss_value is: 0.613272
After 40 trainging steps ,validation accuarcy is 68.4%
After 41 steps, loss_value is: 0.618205
After 41 trainging steps ,validation accuarcy is 67.4%
After 42 steps, loss_value is: 0.619036
After 42 trainging steps ,validation accuarcy is 67.4%
After 43 steps, loss_value is: 0.613694
After 43 trainging steps ,validation accuarcy is 68.8%
After 44 steps, loss_value is: 0.612011
After 44 trainging steps ,validation accuarcy is 69%
After 45 steps, loss_value is: 0.615482
After 45 trainging steps ,validation accuarcy is 68.2%
After 46 steps, loss_value is: 0.615742
After 46 trainging steps ,validation accuarcy is 68.6%
After 47 steps, loss_value is: 0.612004
After 47 trainging steps ,validation accuarcy is 69.2%
After 48 steps, loss_value is: 0.611249
After 48 trainging steps ,validation accuarcy is 68.8%
After 49 steps, loss_value is: 0.613590
After 49 trainging steps ,validation accuarcy is 68%
After 50 steps, loss_value is: 0.613122
After 50 trainging steps ,validation accuarcy is 68.2%
After 51 steps, loss_value is: 0.610477
After 51 trainging steps ,validation accuarcy is 69.6%
After 52 steps, loss_value is: 0.610556
After 52 trainging steps ,validation accuarcy is 69.4%
After 53 steps, loss_value is: 0.612072
After 53 trainging steps ,validation accuarcy is 69.6%
After 54 steps, loss_value is: 0.611105
After 54 trainging steps ,validation accuarcy is 69.4%
After 55 steps, loss_value is: 0.609469
After 55 trainging steps ,validation accuarcy is 69%
After 56 steps, loss_value is: 0.610055
After 56 trainging steps ,validation accuarcy is 69.6%
After 57 steps, loss_value is: 0.610685
After 57 trainging steps ,validation accuarcy is 68.8%
After 58 steps, loss_value is: 0.609479
After 58 trainging steps ,validation accuarcy is 69.6%
After 59 steps, loss_value is: 0.608719
After 59 trainging steps ,validation accuarcy is 68.6%
After 60 steps, loss_value is: 0.609388
After 60 trainging steps ,validation accuarcy is 69.2%
After 61 steps, loss_value is: 0.609323
After 61 trainging steps ,validation accuarcy is 69%
After 62 steps, loss_value is: 0.608320
After 62 trainging steps ,validation accuarcy is 68.8%
After 63 steps, loss_value is: 0.608221
After 63 trainging steps ,validation accuarcy is 70%
After 64 steps, loss_value is: 0.608623
After 64 trainging steps ,validation accuarcy is 69%
After 65 steps, loss_value is: 0.608122
After 65 trainging steps ,validation accuarcy is 69.8%
After 66 steps, loss_value is: 0.607521
After 66 trainging steps ,validation accuarcy is 69.2%
After 67 steps, loss_value is: 0.607717
After 67 trainging steps ,validation accuarcy is 69.6%
After 68 steps, loss_value is: 0.607725
After 68 trainging steps ,validation accuarcy is 69.4%
After 69 steps, loss_value is: 0.607165
After 69 trainging steps ,validation accuarcy is 69.4%
After 70 steps, loss_value is: 0.607002
After 70 trainging steps ,validation accuarcy is 70.6%
After 71 steps, loss_value is: 0.607157
After 71 trainging steps ,validation accuarcy is 69.8%
After 72 steps, loss_value is: 0.606860
After 72 trainging steps ,validation accuarcy is 70.4%
After 73 steps, loss_value is: 0.606500
After 73 trainging steps ,validation accuarcy is 70.4%
After 74 steps, loss_value is: 0.606548
After 74 trainging steps ,validation accuarcy is 69.6%
After 75 steps, loss_value is: 0.606474
After 75 trainging steps ,validation accuarcy is 69.4%
After 76 steps, loss_value is: 0.606132
After 76 trainging steps ,validation accuarcy is 69.8%
After 77 steps, loss_value is: 0.606027
After 77 trainging steps ,validation accuarcy is 70%
After 78 steps, loss_value is: 0.606036
After 78 trainging steps ,validation accuarcy is 70%
After 79 steps, loss_value is: 0.605798
After 79 trainging steps ,validation accuarcy is 70%
After 80 steps, loss_value is: 0.605594
After 80 trainging steps ,validation accuarcy is 70%
After 81 steps, loss_value is: 0.605578
After 81 trainging steps ,validation accuarcy is 69.8%
After 82 steps, loss_value is: 0.605439
After 82 trainging steps ,validation accuarcy is 69.8%
After 83 steps, loss_value is: 0.605217
After 83 trainging steps ,validation accuarcy is 69.8%
After 84 steps, loss_value is: 0.605147
After 84 trainging steps ,validation accuarcy is 70.4%
After 85 steps, loss_value is: 0.605065
After 85 trainging steps ,validation accuarcy is 70%
After 86 steps, loss_value is: 0.604867
After 86 trainging steps ,validation accuarcy is 70.4%
After 87 steps, loss_value is: 0.604748
After 87 trainging steps ,validation accuarcy is 69.6%
After 88 steps, loss_value is: 0.604678
After 88 trainging steps ,validation accuarcy is 70%
After 89 steps, loss_value is: 0.604516
After 89 trainging steps ,validation accuarcy is 69.6%
After 90 steps, loss_value is: 0.604373
After 90 trainging steps ,validation accuarcy is 70%
After 91 steps, loss_value is: 0.604296
After 91 trainging steps ,validation accuarcy is 70.2%
After 92 steps, loss_value is: 0.604165
After 92 trainging steps ,validation accuarcy is 70.2%
After 93 steps, loss_value is: 0.604015
After 93 trainging steps ,validation accuarcy is 69.8%
After 94 steps, loss_value is: 0.603924
After 94 trainging steps ,validation accuarcy is 69.6%
After 95 steps, loss_value is: 0.603809
After 95 trainging steps ,validation accuarcy is 69.8%
After 96 steps, loss_value is: 0.603663
After 96 trainging steps ,validation accuarcy is 69.8%
After 97 steps, loss_value is: 0.603559
After 97 trainging steps ,validation accuarcy is 70.2%
After 98 steps, loss_value is: 0.603452
After 98 trainging steps ,validation accuarcy is 70.2%
After 99 steps, loss_value is: 0.603315
After 99 trainging steps ,validation accuarcy is 70.2%
After 100 steps, loss_value is: 0.603201
After 100 trainging steps ,validation accuarcy is 70%
After 200 steps, loss_value is: 0.590729
After 200 trainging steps ,validation accuarcy is 71.6%
After 300 steps, loss_value is: 0.577428
After 300 trainging steps ,validation accuarcy is 72.2%
After 400 steps, loss_value is: 0.564583
After 400 trainging steps ,validation accuarcy is 74.2%
After 500 steps, loss_value is: 0.552903
After 500 trainging steps ,validation accuarcy is 73.8%
After 600 steps, loss_value is: 0.542685
After 600 trainging steps ,validation accuarcy is 74.6%
After 700 steps, loss_value is: 0.533971
After 700 trainging steps ,validation accuarcy is 75.2%
After 800 steps, loss_value is: 0.526672
After 800 trainging steps ,validation accuarcy is 75%
After 900 steps, loss_value is: 0.520637
After 900 trainging steps ,validation accuarcy is 74.8%
Testing Accuracyis 79.8507%

