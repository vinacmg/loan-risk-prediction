import tensorflow as tf
import model
import numpy as np
import params
import matplotlib
import matplotlib.pyplot as plt

x = np.load('data/x_clean.npy')
y = np.load('data/y_clean.npy')

inputs = tf.placeholder(shape=[None, x.shape[1]],
                        dtype=tf.float32, name='inputs')
targets = tf.placeholder(shape=[None], dtype=tf.float32, name='targets')
hidden_size = params.hidden_size
activation = tf.nn.relu
batch_size = params.batch_size
init_minval = params.init_minval
init_maxval = params.init_maxval
lr = params.lr
optimizer = tf.train.AdamOptimizer(lr)
epochs = params.epochs
training_batch_steps = params.training_batch_steps

ann = model.Model(inputs, targets, hidden_size, activation,
                  batch_size, init_minval, init_maxval, optimizer)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Training...')
loss_track = []

for epoch in range(0, epochs):
    for batch in range(0, training_batch_steps):
        ini_range = batch*batch_size
        end_range = (batch+1)*batch_size
        input_x = x[ini_range:end_range, :]
        input_y = y[ini_range:end_range]
        
        _, l = (sess.run([ann.optimize, ann.loss], feed_dict={
            inputs: input_x,
            targets: input_y
        }))
        
        loss_track.append(np.asscalar(l))

    print('epoch '+ str(epoch) +' end...', end=' ')
    print('loss: '+str(loss_track[-1]))

saver = tf.train.Saver()
saver.save(sess, "save/model")
accuracy = []

for batch in range(training_batch_steps, 722):
    ini_range = batch*batch_size
    end_range = (batch+1)*batch_size
    input_x = x[ini_range:end_range,:]
    input_y = y[ini_range:end_range]

    a = sess.run(ann.accuracy, feed_dict={
        inputs: input_x,
        targets: input_y
    })

    accuracy.append(np.asscalar(a))

print("accuracy = {0}".format(np.mean(accuracy)))
loss_track = np.array(loss_track)
np.save('save/loss.npy', loss_track)
plt.plot(loss_track)
plt.savefig('loss.png', dpi=300)
plt.show()

