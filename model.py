import tensorflow as tf
import functools

def define_scope(function): #read more at https://danijar.com/structuring-your-tensorflow-models/
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class Model():

	def __init__(self, x, y, hidden_size, activation, batch_size, init_minval, init_maxval, optimizer):
		self.x = x
		self.y = y
		self.hidden_size = hidden_size
		self.activation = activation
		self.batch_size = batch_size
		self.init_minval = init_minval
		self.init_maxval = init_maxval
		self.optimizer = optimizer
		self.optimize
		self.prediction
		self.accuracy

	@define_scope
	def core(self):
		print('building model...')
		a1 = tf.contrib.layers.fully_connected(self.x, 
			num_outputs=self.hidden_size,
			activation_fn=self.activation,
			weights_initializer=tf.random_uniform_initializer(self.init_minval, self.init_maxval),
			biases_initializer=tf.constant_initializer(0.0),
			scope='layer1')
		a2 = tf.contrib.layers.fully_connected(a1, 
			num_outputs=1,
			activation_fn=None,
			weights_initializer=tf.random_uniform_initializer(self.init_minval, self.init_maxval),
			biases_initializer=tf.constant_initializer(0.0),
			scope='layer2')

		self._out = a2

		return self._out

	@define_scope
	def loss(self):
		print('setting loss...')
		labels = tf.reshape(self.y, tf.shape(self.core))
		loss =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.core))
		return loss

	@define_scope
	def optimize(self):
		print('setting optimization...')
		return self.optimizer.minimize(self.loss)

	@define_scope
	def prediction(self):
		print('setting prediction...')
		logits = tf.sigmoid(self.core)
		return tf.round(logits)

	@define_scope
	def accuracy(self):
		labels = tf.reshape(self.y, tf.shape(self.core))
		dif = labels - self.prediction
		nonzero = tf.count_nonzero(dif)
		total = self.batch_size
		return (total - nonzero)/total