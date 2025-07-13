import tensorflow as tf

#input1, input2 and bias
train_in = [
    [1., 1.,1],
    [1., 0,1],
    [0, 1.,1],
    [0, 0,1]]
 
#output
train_out = [
[1.],
[0],
[0],
[0]]


#weight variable initialized with random values using random_normal()
w = tf.Variable(tf.random_normal([3, 1], seed=12))

#Placeholder for input and Output
x = tf.placeholder(tf.float32,[None,3])
y = tf.placeholder(tf.float32,[None,1])

#calculate output 
output = tf.nn.relu(tf.matmul(x, w))

#Calculate the Cost or Error:
#Mean Squared Loss or Error
loss = tf.reduce_sum(tf.square(output - y))

#Minimize Error:
#Minimize loss using GradientDescentOptimizer with a learning rate of 0.01
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#Initialize all the variables:
#Initialize all the global variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Training Perceptron learning algorithm in Iterations:
training_epochs = 1000

#Compute cost w.r.t to input vector for 1000 epochs
for epoch in range(training_epochs):
    sess.run(train, {x:train_in,y:train_out})
    cost = sess.run(loss,feed_dict={x:train_in,y:train_out})
    if epoch > 990:
        print('Epoch--',epoch,'--loss--',cost)


##################################################################
## visit scalar topics ML perceptron, do the sonar wali example ##
##################################################################




