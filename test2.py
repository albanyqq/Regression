import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#生成随机数
x_data= np.linspace(-0.5,0.5,500)[:,np.newaxis]
noise =np.random.normal(0,0.02,x_data.shape)
y_date=np.square(x_data)+noise

#定义空白存放位 placehoder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
weight_l1=tf.Variable(tf.random_normal([1,10]))
blases_l1=tf.Variable(tf.zeros([1,10]))
wx_plus_l1=tf.matmul(x,weight_l1)+blases_l1
l1=tf.nn.tanh(wx_plus_l1)

#定义神经网络输出层
weight_l2=tf.Variable(tf.random_normal([10,1]))
blases_l2=tf.Variable(tf.zeros([1,1]))
wx_plus_l2=tf.matmul(l1,weight_l2)+blases_l2
prediction=tf.nn.tanh(wx_plus_l2)

loss =tf.reduce_mean(tf.square(y-prediction))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(train_step,feed_dict={x:x_data,y:y_date})

    prediction_value =sess.run(prediction,feed_dict={x:x_data})

#图像显示一定时间后消失
    plt.ion()

    plt.scatter(x_data, y_date, c='r')
    plt.plot(x_data,prediction_value,'g-',lw=4)
    plt.pause(10)  # 显示秒数
    plt.close()