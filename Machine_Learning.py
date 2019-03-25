import tensorflow as tf
import numpy as np
import csv
import pandas as pd


reader = csv.reader(open("wine.csv"), delimiter=",")
x = list(reader)
Data = np.array(x).astype("float")
m =Data.shape[0]
n = Data.shape[1]


# n  is the columns of the array
print( "columns  " + str(n))
# m is the rows of the array
print( "rows  " + str(m))




#project data
inp= Data

#project success
outs = csv.reader(open("quality.csv"), delimiter=",")
he = list(outs)
out= np.array(he).astype("float")

inp=np.array(inp)
out=np.array(out)


inputs=tf.placeholder('float',[None,n],name='Input')
targets=tf.placeholder('float',name='Target')

weight1=tf.Variable(tf.random_normal(shape=[11,3],stddev=0.1),name="Weight1")
biases1=tf.Variable(tf.random_normal(shape=[3],stddev=0.1),name="Biases1")

#this is for tensor board to anal;yse= data
tf.summary.histogram("weight_1",weight1)

hLayer=tf.matmul(inputs,weight1)
hLayer=hLayer+biases1

hLayer=tf.sigmoid(hLayer, name='hActivation')


weight2=tf.Variable(tf.random_normal(shape=[3,11],stddev=0.1),name="Weight2")
biases2=tf.Variable(tf.random_normal(shape=[11],stddev=0.1),name="Biases2")

#also for tensor board
tf.summary.histogram("weight_2",weight2)

output=tf.matmul(hLayer,weight2)
output=output+biases2
output=tf.nn.softplus(output, name='outActivation')


#cost=tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=targets)
# update  tf.nn.softmax_cross_entropy_with_logits is for classification problems only
# we will be using tf.squared_difference()
cost=tf.squared_difference(targets, output)
cost=tf.reduce_mean(cost)


#also for tensorboard
tf.summary.scalar("cost", cost)
#tf.train.GradientDescentOptimizer.minimize()
optimizer=tf.train.AdamOptimizer().minimize(cost)


epochs=10000 # number of time we want to repeat

#within the session we also write tensorboard code.
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        error,_ =sess.run([cost,optimizer],feed_dict={inputs: inp,targets:out})
        print(i,error)
        print(Data[:,0])


    while True:
        a = input("type 1st input :")
        o = input("type 2nd input :")
        b = input("type 3rd input :")
        c = input("type 4th input :")
        d = input("type 5th input :")
        e = input("type 6th input :")
        f = input("type 7th input :")
        g = input("type 8th input :")
        h = input("type 9th input :")
        i = input("type 10th input :")
        j = input("type 11th input :")

        inp=[[a,o,b,c,d,e,f,g,h,i,j]]
        inp=np.array(inp)
        prediction=sess.run([cost],feed_dict={inputs: inp})
        print(prediction)
