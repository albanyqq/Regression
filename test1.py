import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

#随机生产点
num_points =500
vectors_set =[]
for i in range(num_points):
    x1 =np.random.normal(0.0,0.60)
    y1 =x1*0.1+0.28+np.random.normal(0.0,0.02)
    vectors_set.append([x1,y1])

#生成样本
x_date =[v[0] for v in vectors_set]
y_date =[v[1] for v in vectors_set]




#设定计算公式 y=wx+b
w = tf.Variable(tf.random_uniform([1],-1,1),name ='w')
b =tf.Variable(tf.zeros([1]),name='b')

y=w*x_date+b

#预估值和实际值点均方误差作为损失
loss=tf.reduce_mean(tf.square(y-y_date),name ='loss')
#梯度下降 0.2为学习率   就是每次调整的步长
optimizer =tf.train.GradientDescentOptimizer(0.3)
#最小化误差值
train =optimizer.minimize(loss,name ='train')

sess =tf.Session()
#初始化操作
init =tf.global_variables_initializer()
sess.run(init)


for step in range(30):
    time.sleep(1)
    sess.run(train)
    #图像显示一定时间后消失
    plt.ion()

    plt.scatter(x_date, y_date, c='r')
    plt.plot(x_date,sess.run(w)*x_date+sess.run(b))

    plt.pause(0.5)  # 显示秒数
    plt.close()
    #print("w=",sess.run(w),"b=",sess.run(b),"loss=",sess.run(loss))

