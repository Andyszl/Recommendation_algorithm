import tensorflow as tf
import  numpy as np
import pandas as pd
from tensorflow.python.framework import ops 
#ops.reset_default_graph()

from sklearn.preprocessing import  OneHotEncoder
import pandas as pd
#参数设定
#特征数量
n_features = 8
#label个数
n_class = 2

#定义训练轮数
#training_steps = 1000
#学习率
learning_rate=0.01


#定义训练轮数
training_steps = 1000
#学习率
#learning_rate=0.1
#隐层K
fv=20

dnn_layer=[64,32]
dnn_active_fuc=['relu','relu','relu']


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


def udf_full_connect(Input,input_size,output_size,activation='relu'):
    #生成或获取weights和biases
    weights=tf.get_variable("weights",[input_size,output_size],initializer=tf.glorot_normal_initializer(),trainable=True)
    biases=tf.get_variable("biases",[output_size],initializer=tf.glorot_normal_initializer(),trainable=True)
    
    #全链接 
    layer=tf.matmul(Input,weights)+biases
    if activation=="relu":
        layer=tf.nn.relu(layer)
    elif activation=="tanh":
        layer=tf.nn.tanh(layer)
        
    return layer
    
    
ops.reset_default_graph()
with tf.name_scope("Input"):
    x = tf.placeholder(tf.float32, [None, n_features])
    y = tf.placeholder(tf.float32, [None, n_class])
    Input_x = tf.reshape(x, shape=[-1, n_features, 1]) # None * feature_size 
    print("Input_x",Input_x)
    

# 模型参数parameter
with tf.name_scope("Parameter"):
    W = tf.Variable(tf.zeros([n_features, n_class]),name="w")
    b = tf.Variable(tf.zeros([n_class]),name="b")
    v = tf.Variable(tf.zeros([n_features, fv]),name="V")
    embeddings = tf.multiply(v, Input_x) # None * V * X 



    # 定义模型，此处使用与线性回归一样的定义
    # 因为在后面定义损失的时候会加上映射
with tf.name_scope("Prediction"):
    
    Y_liner = tf.matmul(x, W) + b
    #0.5*((sum(v*x))^2 - sum((v*x)^2)) 
    summed_features_emb = tf.reduce_sum(embeddings, 1)  # sum(v*x)
    summed_features_emb_square = tf.square(summed_features_emb)  # (sum(v*x))^2

    # square_sum part
    squared_features_emb = tf.square(embeddings) # (v*x)^2
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)   # sum((v*x)^2)

    
    Y_pair = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # 0.5*((sum(v*x))^2 - sum((v*x)^2))
    
    
    pred= tf.concat([Y_liner, Y_pair], axis=1) 
    
""" 3 Deep层网络输出 """
print("3 Deep层网络输出" )
with tf.name_scope("Deep"):
    # 第一层计算
    print("lay%s, input_size: %s, output_size: %s, active_fuc: %s" % (1, n_features*fv, dnn_layer[0], dnn_active_fuc[0]))
    with tf.variable_scope("deep_layer1", reuse=tf.AUTO_REUSE):
        input_size = n_features*fv
        output_size = dnn_layer[0]
        deep_inputs = tf.reshape(embeddings, shape=[-1, input_size]) # None * (F*K)
        print("%s: %s" % ("lay1, deep_inputs", deep_inputs))
       
        # 全连接计算    
        deep_outputs = udf_full_connect(deep_inputs, input_size, output_size, dnn_active_fuc[0])
        print("%s: %s" % ("lay1, deep_outputs", deep_outputs))
        # batch_norm
        #if is_batch_norm:
        #    deep_outputs = tf.layers.batch_normalization(deep_outputs, axis=-1, training=is_train) 
        # 输出dropout
        #if is_train and is_dropout_dnn:
        #    deep_outputs = tf.nn.dropout(deep_outputs, dropout_dnn[1])
    # 中间层计算
    
    for i in range(len(dnn_layer) - 1):
        with tf.variable_scope("deep_layer%d"%(i+2), reuse=tf.AUTO_REUSE):
            print("lay%s, input_size: %s, output_size: %s, active_fuc: %s" % (i+2, dnn_layer[i], dnn_layer[i+1], dnn_active_fuc[i+1]))
            # 全连接计算
            deep_outputs = udf_full_connect(deep_outputs, dnn_layer[i], dnn_layer[i+1], dnn_active_fuc[i+1])
            print("lay%s, deep_outputs: %s" % (i+2, deep_outputs))
            # batch_norm
            #if is_batch_norm:
            #    deep_outputs = tf.layers.batch_normalization(deep_outputs, axis=-1, training=is_train)
            # 输出dropout  
           # if is_train and is_dropout_dnn:
             #   deep_outputs = tf.nn.dropout(deep_outputs, dropout_dnn[i+2])
             
    # 输出层计算
    print("lay_last, input_size: %s, output_size: %s, active_fuc: %s" % (dnn_layer[-1], 2, dnn_active_fuc[-1]))
    with tf.variable_scope("deep_layer%d"%(len(dnn_layer)+1), reuse=tf.AUTO_REUSE):
        deep_outputs = udf_full_connect(deep_outputs, dnn_layer[-1],2, dnn_active_fuc[-1])
        print("lay_last, deep_outputs: %s" % (deep_outputs))

    # 正则化，默认L2
    dnn_regularization = 0.0
    for j in range(len(dnn_layer)+1):        
        with tf.variable_scope("deep_layer%d"%(j+1), reuse=True):
            weights = tf.get_variable("weights")
            dnn_regularization = dnn_regularization + tf.nn.l2_loss(weights)

Y_deep=deep_outputs    
concat_input = tf.concat([Y_liner, Y_pair, Y_deep], axis=1)    
Y_sum = tf.reduce_sum(concat_input, 1)
print("Y_sum",Y_sum) 
score=tf.nn.sigmoid(Y_sum,name='score')
#score=tf.reshape(score, shape=[-1, 1])
 
    
    
    
    
# 定义损失函数
with tf.name_scope("losses"):
    with tf.name_scope("error_loss"):
        print("pred",tf.shape(Y_sum))
        print("y",tf.shape(y))
        error_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(Y_sum, [-1]), labels=tf.reshape(tf.cast(tf.argmax(y,axis=1),tf.float32), [-1]))) 

    tf.add_to_collection("losses", error_loss)      #加入集合的操作

    #在权重参数上实现L2正则化
    with tf.name_scope("regularization"):
        regularizer = tf.contrib.layers.l2_regularizer(0.01)
        regularization = regularizer(W)+regularizer(v)+dnn_regularization
    tf.add_to_collection("losses",regularization)     #加入集合的操作

    #get_collection()函数获取指定集合中的所有个体，这里是获取所有损失值
    #并在add_n()函数中进行加和运算
    loss = tf.add_n(tf.get_collection("losses"))

#定义一个优化器，学习率为固定为0.01，注意在实际应用中这个学习率数值应该大于0.01
with tf.name_scope("Train"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 准确率
with tf.name_scope("accuracy"):

    #correct_prediction = tf.equal(tf.argmax(score, axis=1), tf.argmax(y, axis=1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.metrics.auc(tf.argmax(y, axis=1), score)
    tf.summary.histogram("accuracy",accuracy)
    #tf.summary.scalar("accuracy",accuracy)

merged=tf.summary.merge_all()

with tf.Session() as sess:

    tf.global_variables_initializer().run()
    sess.run(tf.local_variables_initializer())
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
            print("After %d trainging steps ,validation accuarcy is %g%%"%(i,voliadata_accuracy[0]*100))
        #xs,ys =data.train.next_batch(200)
        #sess.run(train_op,feed_dict={x:xs,y:ys})
  
    print("Testing Accuracyis %g%%"%(accuracy[0].eval({x: X_test, y:y_test})*100))
writer.close()


out:
Input_x Tensor("Input/Reshape:0", shape=(?, 8, 1), dtype=float32)
3 Deep层网络输出
lay1, input_size: 160, output_size: 64, active_fuc: relu
lay1, deep_inputs: Tensor("Deep/deep_layer1/Reshape:0", shape=(?, 160), dtype=float32)
lay1, deep_outputs: Tensor("Deep/deep_layer1/Relu:0", shape=(?, 64), dtype=float32)
lay2, input_size: 64, output_size: 32, active_fuc: relu
lay2, deep_outputs: Tensor("Deep/deep_layer2/Relu:0", shape=(?, 32), dtype=float32)
lay_last, input_size: 32, output_size: 2, active_fuc: relu
lay_last, deep_outputs: Tensor("Deep/deep_layer3/Relu:0", shape=(?, 2), dtype=float32)
Y_sum Tensor("Sum:0", shape=(?,), dtype=float32)
pred Tensor("losses/error_loss/Shape:0", shape=(1,), dtype=int32)
y Tensor("losses/error_loss/Shape_1:0", shape=(2,), dtype=int32)
After 0 steps, loss_value is: 45.111355
After 0 trainging steps ,validation accuarcy is 0%
After 1 steps, loss_value is: 86.908089
After 1 trainging steps ,validation accuarcy is 41.8757%
After 2 steps, loss_value is: 40.317459
After 2 trainging steps ,validation accuarcy is 48.262%
After 3 steps, loss_value is: 33.523735
After 3 trainging steps ,validation accuarcy is 50.7288%
After 4 steps, loss_value is: 27.494133
After 4 trainging steps ,validation accuarcy is 50.1347%
After 5 steps, loss_value is: 33.555344
After 5 trainging steps ,validation accuarcy is 51.4486%
After 6 steps, loss_value is: 21.347748
After 6 trainging steps ,validation accuarcy is 52.1772%
After 7 steps, loss_value is: 15.066051
After 7 trainging steps ,validation accuarcy is 52.8318%
After 8 steps, loss_value is: 10.888313
After 8 trainging steps ,validation accuarcy is 52.721%
After 9 steps, loss_value is: 14.330173
After 9 trainging steps ,validation accuarcy is 53.9007%
After 10 steps, loss_value is: 12.996469
After 10 trainging steps ,validation accuarcy is 54.5017%
After 11 steps, loss_value is: 11.106708
After 11 trainging steps ,validation accuarcy is 54.4249%
After 12 steps, loss_value is: 11.262444
After 12 trainging steps ,validation accuarcy is 54.3352%
After 13 steps, loss_value is: 6.987972
After 13 trainging steps ,validation accuarcy is 54.8974%
After 14 steps, loss_value is: 11.056909
After 14 trainging steps ,validation accuarcy is 55.7321%
After 15 steps, loss_value is: 7.565862
After 15 trainging steps ,validation accuarcy is 55.2047%
After 16 steps, loss_value is: 10.160446
After 16 trainging steps ,validation accuarcy is 54.9933%
After 17 steps, loss_value is: 8.466215
After 17 trainging steps ,validation accuarcy is 55.5308%
After 18 steps, loss_value is: 6.560686
After 18 trainging steps ,validation accuarcy is 56.1716%
After 19 steps, loss_value is: 6.813130
After 19 trainging steps ,validation accuarcy is 55.9793%
After 20 steps, loss_value is: 5.249652
After 20 trainging steps ,validation accuarcy is 55.857%
After 21 steps, loss_value is: 5.865858
After 21 trainging steps ,validation accuarcy is 56.7342%
After 22 steps, loss_value is: 4.467908
After 22 trainging steps ,validation accuarcy is 57.4584%
After 23 steps, loss_value is: 5.078068
After 23 trainging steps ,validation accuarcy is 57.7768%
After 24 steps, loss_value is: 3.755702
After 24 trainging steps ,validation accuarcy is 57.9369%
After 25 steps, loss_value is: 4.746198
After 25 trainging steps ,validation accuarcy is 58.708%
After 26 steps, loss_value is: 3.099093
After 26 trainging steps ,validation accuarcy is 59.3493%
After 27 steps, loss_value is: 4.394212
After 27 trainging steps ,validation accuarcy is 59.8457%
After 28 steps, loss_value is: 2.474697
After 28 trainging steps ,validation accuarcy is 59.8904%
After 29 steps, loss_value is: 4.516461
After 29 trainging steps ,validation accuarcy is 60.435%
After 30 steps, loss_value is: 2.335449
After 30 trainging steps ,validation accuarcy is 60.7595%
After 31 steps, loss_value is: 3.705245
After 31 trainging steps ,validation accuarcy is 61.2026%
After 32 steps, loss_value is: 2.124532
After 32 trainging steps ,validation accuarcy is 61.2954%
After 33 steps, loss_value is: 3.311386
After 33 trainging steps ,validation accuarcy is 61.7497%
After 34 steps, loss_value is: 2.075816
After 34 trainging steps ,validation accuarcy is 62.0967%
After 35 steps, loss_value is: 3.003377
After 35 trainging steps ,validation accuarcy is 62.492%
After 36 steps, loss_value is: 1.828163
After 36 trainging steps ,validation accuarcy is 62.6013%
After 37 steps, loss_value is: 2.864039
After 37 trainging steps ,validation accuarcy is 62.9892%
After 38 steps, loss_value is: 1.865129
After 38 trainging steps ,validation accuarcy is 63.2321%
After 39 steps, loss_value is: 2.344306
After 39 trainging steps ,validation accuarcy is 63.496%
After 40 steps, loss_value is: 1.842699
After 40 trainging steps ,validation accuarcy is 63.6694%
After 41 steps, loss_value is: 1.931493
After 41 trainging steps ,validation accuarcy is 64.0086%
After 42 steps, loss_value is: 1.830762
After 42 trainging steps ,validation accuarcy is 64.3336%
After 43 steps, loss_value is: 1.796602
After 43 trainging steps ,validation accuarcy is 64.5413%
After 44 steps, loss_value is: 1.666537
After 44 trainging steps ,validation accuarcy is 64.746%
After 45 steps, loss_value is: 1.511870
After 45 trainging steps ,validation accuarcy is 65.0429%
After 46 steps, loss_value is: 1.688789
After 46 trainging steps ,validation accuarcy is 65.3172%
After 47 steps, loss_value is: 1.386824
After 47 trainging steps ,validation accuarcy is 65.4934%
After 48 steps, loss_value is: 1.693570
After 48 trainging steps ,validation accuarcy is 65.7139%
After 49 steps, loss_value is: 1.328331
After 49 trainging steps ,validation accuarcy is 65.951%
After 50 steps, loss_value is: 1.444596
After 50 trainging steps ,validation accuarcy is 66.1579%
After 51 steps, loss_value is: 1.349614
After 51 trainging steps ,validation accuarcy is 66.3254%
After 52 steps, loss_value is: 1.220221
After 52 trainging steps ,validation accuarcy is 66.5412%
After 53 steps, loss_value is: 1.380549
After 53 trainging steps ,validation accuarcy is 66.7424%
After 54 steps, loss_value is: 1.240671
After 54 trainging steps ,validation accuarcy is 66.8698%
After 55 steps, loss_value is: 1.164050
After 55 trainging steps ,validation accuarcy is 67.0415%
After 56 steps, loss_value is: 1.265618
After 56 trainging steps ,validation accuarcy is 67.2139%
After 57 steps, loss_value is: 1.127204
After 57 trainging steps ,validation accuarcy is 67.3496%
After 58 steps, loss_value is: 1.133717
After 58 trainging steps ,validation accuarcy is 67.5273%
After 59 steps, loss_value is: 1.151599
After 59 trainging steps ,validation accuarcy is 67.7009%
After 60 steps, loss_value is: 1.051734
After 60 trainging steps ,validation accuarcy is 67.8405%
After 61 steps, loss_value is: 1.069535
After 61 trainging steps ,validation accuarcy is 67.9958%
After 62 steps, loss_value is: 1.088805
After 62 trainging steps ,validation accuarcy is 68.1519%
After 63 steps, loss_value is: 1.016789
After 63 trainging steps ,validation accuarcy is 68.2784%
After 64 steps, loss_value is: 0.998553
After 64 trainging steps ,validation accuarcy is 68.4339%
After 65 steps, loss_value is: 1.026862
After 65 trainging steps ,validation accuarcy is 68.5892%
After 66 steps, loss_value is: 0.983794
After 66 trainging steps ,validation accuarcy is 68.704%
After 67 steps, loss_value is: 0.939855
After 67 trainging steps ,validation accuarcy is 68.8406%
After 68 steps, loss_value is: 0.961708
After 68 trainging steps ,validation accuarcy is 68.9607%
After 69 steps, loss_value is: 0.984688
After 69 trainging steps ,validation accuarcy is 69.0687%
After 70 steps, loss_value is: 0.935883
After 70 trainging steps ,validation accuarcy is 69.1804%
After 71 steps, loss_value is: 0.892686
After 71 trainging steps ,validation accuarcy is 69.2861%
After 72 steps, loss_value is: 0.896821
After 72 trainging steps ,validation accuarcy is 69.3994%
After 73 steps, loss_value is: 0.915590
After 73 trainging steps ,validation accuarcy is 69.5052%
After 74 steps, loss_value is: 0.913737
After 74 trainging steps ,validation accuarcy is 69.5942%
After 75 steps, loss_value is: 0.870147
After 75 trainging steps ,validation accuarcy is 69.6798%
After 76 steps, loss_value is: 0.834181
After 76 trainging steps ,validation accuarcy is 69.7648%
After 77 steps, loss_value is: 0.819087
After 77 trainging steps ,validation accuarcy is 69.8564%
After 78 steps, loss_value is: 0.821339
After 78 trainging steps ,validation accuarcy is 69.9469%
After 79 steps, loss_value is: 0.821229
After 79 trainging steps ,validation accuarcy is 70.0324%
After 80 steps, loss_value is: 0.807020
After 80 trainging steps ,validation accuarcy is 70.1175%
After 81 steps, loss_value is: 0.791589
After 81 trainging steps ,validation accuarcy is 70.197%
After 82 steps, loss_value is: 0.770755
After 82 trainging steps ,validation accuarcy is 70.2717%
After 83 steps, loss_value is: 0.754591
After 83 trainging steps ,validation accuarcy is 70.3484%
After 84 steps, loss_value is: 0.742554
After 84 trainging steps ,validation accuarcy is 70.4239%
After 85 steps, loss_value is: 0.733897
After 85 trainging steps ,validation accuarcy is 70.4959%
After 86 steps, loss_value is: 0.728986
After 86 trainging steps ,validation accuarcy is 70.5617%
After 87 steps, loss_value is: 0.727839
After 87 trainging steps ,validation accuarcy is 70.6228%
After 88 steps, loss_value is: 0.740576
After 88 trainging steps ,validation accuarcy is 70.6825%
After 89 steps, loss_value is: 0.765212
After 89 trainging steps ,validation accuarcy is 70.7353%
After 90 steps, loss_value is: 0.821501
After 90 trainging steps ,validation accuarcy is 70.7884%
After 91 steps, loss_value is: 0.858191
After 91 trainging steps ,validation accuarcy is 70.8305%
After 92 steps, loss_value is: 0.904651
After 92 trainging steps ,validation accuarcy is 70.8616%
After 93 steps, loss_value is: 0.912382
After 93 trainging steps ,validation accuarcy is 70.891%
After 94 steps, loss_value is: 0.883997
After 94 trainging steps ,validation accuarcy is 70.9134%
After 95 steps, loss_value is: 0.814067
After 95 trainging steps ,validation accuarcy is 70.9405%
After 96 steps, loss_value is: 0.732851
After 96 trainging steps ,validation accuarcy is 70.9724%
After 97 steps, loss_value is: 0.665132
After 97 trainging steps ,validation accuarcy is 71.0125%
After 98 steps, loss_value is: 0.633186
After 98 trainging steps ,validation accuarcy is 71.0563%
After 99 steps, loss_value is: 0.620577
After 99 trainging steps ,validation accuarcy is 71.102%
After 100 steps, loss_value is: 0.622833
After 100 trainging steps ,validation accuarcy is 71.1457%
After 200 steps, loss_value is: 0.601904
After 200 trainging steps ,validation accuarcy is 72.6455%
After 300 steps, loss_value is: 0.514871
After 300 trainging steps ,validation accuarcy is 74.3373%
After 400 steps, loss_value is: 0.525866
After 400 trainging steps ,validation accuarcy is 75.5403%
After 500 steps, loss_value is: 0.517595
After 500 trainging steps ,validation accuarcy is 76.3901%
After 600 steps, loss_value is: 0.505924
After 600 trainging steps ,validation accuarcy is 76.9978%
After 700 steps, loss_value is: 0.490292
After 700 trainging steps ,validation accuarcy is 77.4537%
After 800 steps, loss_value is: 0.489119
After 800 trainging steps ,validation accuarcy is 77.8268%
After 900 steps, loss_value is: 0.582229
After 900 trainging steps ,validation accuarcy is 78.096%
Testing Accuracyis 78.287%
