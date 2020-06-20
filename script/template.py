PATH_TO_DATASET_DIR=''
#Expected structure:
# - code.py
# - (DATASETDIR)/dale/dale.h5
# - (DATASETDIR)/dale/dale.h5
#Datasets provided could go offline, refer to official sites:
# https://jack-kelly.com/data/
# http://redd.csail.mit.edu/
#    ^*(needs conversion to HDF5)
#------------------------------------------------#
#                    Import                      #
#------------------------------------------------#

import tensorflow as tf
from tensorflow import print
import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, Dense, Input, add
from tensorflow_addons.layers import WeightNormalization
from tensorflow_addons.metrics import HammingLoss, F1Score
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.train import ExponentialMovingAverage
from tensorflow.keras.losses import MSE, BinaryCrossentropy
from tensorflow.keras.metrics import Mean, BinaryAccuracy
from tensorflow.compat.v1 import assign, enable_eager_execution
from tensorflow.keras.backend import get_value
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nilmtk import DataSet
from gdown import download
from matplotlib import pyplot as plt
from tensorflow import print, where, zeros_like
from tensorflow.math import is_nan

#------------------------------------------------#

#------------------------------------------------#
#                   Download                     #
#------------------------------------------------#
## Dataset directory
os.makedirs(PATH_TO_DATASET_DIR, exist_ok=True)

# DALE h5
os.makedirs(PATH_TO_DATASET_DIR+'dale', exist_ok=True)
if not os.path.exists(os.path.join(PATH_TO_DATASET_DIR,'dale/ukdale.h5')):
	if not os.path.exists('./ukdale.h5.zip'):
		download('https://drive.google.com/uc?id=1mnEQrRrL8SxXwejbhVcrCUglQyKcVkLv', './ukdale.h5.zip', quiet=False)
	cmd="unzip -n 'ukdale.h5.zip' -d {0}'dale/'".format(PATH_TO_DATASET_DIR)
	os.system(cmd)

# REDD h5
os.makedirs(PATH_TO_DATASET_DIR+'redd', exist_ok=True)
if not os.path.exists(os.path.join(PATH_TO_DATASET_DIR,'redd/redd.h5')):
	if not os.path.exists('./redd.h5.zip'):
		download('https://drive.google.com/uc?id=1hn7GPwrtwblTSV8gcrgT3E6rsGDxXPQF', './redd.h5.zip', quiet=False)
	cmd="unzip -n 'redd.h5.zip' -d {0}'redd/'".format(PATH_TO_DATASET_DIR)
	os.system(cmd)

#------------------------------------------------#

#------------------------------------------------#
#                  House Class                   #
#------------------------------------------------#

class house():
	def __init__(self, dataset=None, appliances=None, n=None, window_start=None, window_end=None, main=None, on=None,pwr=None,avg_pwr=None):
		super(house, self).__init__()
		if main is not None and on is not None and pwr is not None and avg_pwr is not None:
			self.main = main
			self.on = on
			self.pwr = pwr
			self.avg_pwr = avg_pwr
		else:
			if window_start is not None:
				dataset.set_window(start=window_start, end=window_end)

			#load labels, 1 label per 32 minutes
			self.on = np.transpose(np.stack(([next(dataset.buildings[n].elec[a].when_on(sample_period=30*64)).to_numpy(dtype=int) for a in appliances])))
			#drop last one as it will match the incomplete 64 sample from main
			self.on=self.on[:-1]

			try:
				self.main = np.stack(next(dataset.buildings[n].elec.mains().load(physical_quantity='power', ac_type='active',sample_period=30)).to_numpy())
				offload=np.squeeze(np.stack(list(tf.data.Dataset.from_tensor_slices(self.main).batch(64,drop_remainder=True))))
				self.main=offload
				self.main = self.main[:-1, :]
			except:
				# fallback to apparent (REDD)
				self.main = np.stack(next(dataset.buildings[n].elec.mains().load(physical_quantity='power', ac_type='apparent',sample_period=30)).to_numpy())
				self.main = self.main[:-1, :]
				offload=np.squeeze(np.stack(list(tf.data.Dataset.from_tensor_slices(self.main).batch(64,drop_remainder=True))))
				self.main=offload
				pass
			try:
				self.pwr = np.transpose(np.stack(([np.squeeze(next(dataset.buildings[n].elec[a].load(physical_quantity='power', ac_type='active',sample_period=30*64)).to_numpy()) for a in appliances])))
				# drop last one as it will match the incomplete 64 sample from main
				self.pwr=self.pwr[:-1]
			except:
				# fallback to apparent (REDD)
				self.pwr = np.transpose(np.stack(([np.squeeze(next(dataset.buildings[n].elec[a].load(physical_quantity='power', ac_type='apparent', sample_period=30*64)).to_numpy()) for a in appliances])))
				# drop last two as it will match the incomplete 64 sample from main
				self.pwr = self.pwr[:-2, :]
				pass
			try:
				self.avg_pwr=np.concatenate([dataset.buildings[n].elec[a].average_energy_per_period(pd.DateOffset(seconds=30)) for a in appliances])
			except Exception as ex:
				print(ex)
			if dataset.metadata['name'] == 'REDD':
				# no kettle in REDD
				self.on = np.concatenate((np.zeros((self.on.shape[0], 1)), self.on), axis=1)
				self.pwr = np.concatenate((np.zeros((self.pwr.shape[0], 1)), self.pwr), axis=1)
				self.avg_pwr = np.concatenate((np.zeros((1)), self.avg_pwr), axis=0)
				# ...REDD
				if n==3:
					self.on=self.on[:-1]
					self.pwr=self.pwr[:-1]

	def toSetGenerator(self, batch_size=1024, expand=True, ratio=0.2, drop_probability=0,scale=True):
		x=self.main
		y=self.on
		z=self.pwr
		if scale:
			s = MinMaxScaler()
			x=s.fit_transform(x)
		if expand:
			x=np.expand_dims(x,axis=1)
			y = np.expand_dims(y, axis=1)
			z =np.expand_dims(z,axis=1)


		if ratio:
			train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=ratio, shuffle=False)
			if not (test_x.shape[0]>=batch_size):
				raise ValueError("\n\nDataset too small to use minibatching at "+str(batch_size)+"\nConsider running it again with batch_size less than "+str(test_x.shape[0])+"\n\n") 
			_n,z=train_test_split((z),test_size=ratio,shuffle=False)
			if drop_probability!=0:
				train_y=self._dropLabels(train_y,drop_probability)
			#Make datasets
			train = tf.data.Dataset.from_tensor_slices((train_x,train_y)).batch(batch_size,drop_remainder=True)
			test = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size, drop_remainder=True)
			power= tf.data.Dataset.from_tensor_slices((z)).batch(batch_size,drop_remainder=True)
			return train,test,power
		else:
			if drop_probability!=0:
				y=self._dropLabels(y,drop_probability)
			#Make datasets
			power= tf.data.Dataset.from_tensor_slices((z)).batch(batch_size,drop_remainder=True)
			train=tf.data.Dataset.from_tensor_slices((x,y)).batch(batch_size,drop_remainder=True)
			return train, power

	def _dropLabels(self, array, probability=0):
		t = np.where(np.random.uniform(0, 1, size=array.shape[0]) < probability, 1, 0)
		copy = np.copy(array)
		copy[t == 1] = -1
		return copy

def condense(hlist=list()):
	# maybe normalize here
	assert(type(hlist) is list)
	scale=False
	s = MinMaxScaler()
	if scale:
		main = np.concatenate(([s.fit_transform(x.main) for x in hlist]))
	else:
		main = np.concatenate(([x.main for x in hlist]))
	on = np.concatenate(([x.on for x in hlist]))
	pwr = np.concatenate(([x.pwr for x in hlist]))
	avg_pwr=np.average([x.avg_pwr for x in hlist],axis=0)
	return house(main=main, on=on, pwr=pwr, avg_pwr=avg_pwr)

#------------------------------------------------#

#------------------------------------------------#
#                     Model                      #
#------------------------------------------------#

class Residual(Model):
	def __init__(self, k, d, normalized=True):
		super(Residual, self).__init__()
		if normalized:
			self.causal1D = WeightNormalization(Conv1D(kernel_size=k,
													   padding="causal",
													   dilation_rate=2 ** (d - 1),
													   activation="relu",
													   filters=64,
													   kernel_initializer='random_normal'))
		else:
			self.causal1D = Conv1D(kernel_size=k,
								   padding="causal",
								   dilation_rate=2 ** (d - 1),
								   activation="relu",
								   filters=64,
								   kernel_initializer='random_normal')
		self.drop = SpatialDropout1D(0.1)
		self.matching_1D = Conv1D(kernel_size=1,
								  padding="same",
								  activation="relu",
								  filters=64,
								  kernel_initializer='random_normal')

	def call(self, input_tensor):
		x = self.causal1D(input_tensor)
		x = self.drop(x)
		x = self.causal1D(x)
		x = self.drop(x)
		return add([x, self.matching_1D(input_tensor)])


class TCN(Model):
	def __init__(self, normalized=True):
		super(TCN, self).__init__()
		self.res1 = Residual(3, 1, normalized)
		self.res2 = Residual(3, 2, normalized)
		self.res3 = Residual(3, 3, normalized)
		self.d1 = Dense(5, activation='sigmoid')

	def call(self, x):
		x = self.res1(x)
		x = self.res2(x)
		x = self.res3(x)
		x = self.d1(x)
		return x

def filter_weights(norm_w):
	out = np.concatenate((norm_w.trainable_weights[1:3], norm_w.trainable_weights[-2:]))
	return out


def transfer_weights(src, dest, averager):
	averager.apply(src.trainable_weights)
	for residual in zip(dest.layers[:-1], src.layers[:-1]):
		residual[0].set_weights([get_value(y) for y in [averager.average(x) for x in filter_weights(residual[1])]])
	dest.layers[-1].set_weights(
		[get_value(y) for y in [averager.average(x) for x in src.layers[-1].trainable_weights]])

#------------------------------------------------#

#------------------------------------------------#
#                   Functions                    #
#------------------------------------------------#
def add_noise(array, sigma=0.1):
	if array is not None:
		noise=np.random.normal(scale=sigma,size=array.shape)
		out=array+noise
		#out = array + np.random.normal(scale=sigma, size=array.shape[0])
		# out=array.copy+np.random.normal(scale=sigma,size=array.shape[0])
		return out
	else:
		print("Skipping empty array")
		return None

def custom_loss(s_pred, t_pred, labels, epoch):
	w = 50 * np.exp(-5 * (1 - (epoch / 80)) ** 2)
	bce = BinaryCrossentropy()
	# ignores values set to -1
	mask=np.where(labels[:,0]>=0,1,0)
	ll = bce(y_pred=s_pred, y_true=labels, sample_weight=mask)
	ll=ll/(len(s_pred[-1]))
	if t_pred is not None:
		lu = MSE(s_pred, t_pred) / (len(s_pred[-1]))
		return tf.math.reduce_mean(ll) + w*tf.math.reduce_mean(lu)
	else:
		return tf.math.reduce_mean(ll)
def ANE(pred, avg_pwr, real_pwr):
	# needs array of average_power -> nilmtk.electric.average_energy_per_period(pandas timeseries.offset_alias)
	avg_pwr = np.cast['float32'](np.squeeze(avg_pwr))
	real_pwr = np.cast['float32'](np.squeeze(real_pwr))
	pred_pwr = np.multiply(np.cast['float32'](pred), avg_pwr)
	x = tf.reduce_sum(pred_pwr, 0)
	y = tf.reduce_sum(real_pwr, 0)
	res=(abs(y - x) / y)
	res = np.where(np.isinf(res), 1, res)
	return res
class early():
	best = 0.0
	p = 0
	l = list()
	initial_p = 0
	e = 0

	def __init__(self, patience, initial_epoch=0):
		self.p = patience
		self.initial_p = patience
		self.e = initial_epoch

	def stop(self, loss, epoch):
		if epoch < self.e:
			return False
		else:
			if not self.l:
				self.best = loss

			self.l.append(loss)
			if loss >= self.best:
				self.p -= 1
				if self.p <= 0:
					return True
				else:
					return False
			else:
				self.best = loss
				self.p = self.initial_p
				return False

def train_step(samples, labels, epoch,with_teacher):
	with tf.GradientTape() as s_tape, tf.GradientTape() as t_tape:
		s_predictions = student(add_noise(samples), training=True)
		if with_teacher:
			t_predictions = teacher(add_noise(samples), training=False)
			loss = custom_loss(s_predictions, t_predictions, labels, epoch)
		else:
			loss = custom_loss(s_predictions, None, labels, epoch)
	s_gradients = s_tape.gradient(loss, student.trainable_weights)
	#t_gradients = t_tape.gradient(loss, teacher.trainable_weights)
	s_optimizer.apply_gradients(zip(s_gradients, student.trainable_weights))
	if with_teacher:
		transfer_weights(student, teacher, ema)
	train_loss(loss)
	train_accuracy(labels, s_predictions)

def test_step(samples, labels, epoch, with_teacher):
	s_predictions = student(add_noise(samples), training=False)
	if with_teacher:
		t_predictions = teacher(add_noise(samples), training=False)
		loss = custom_loss(s_predictions, t_predictions, labels, epoch)

	loss = custom_loss(s_predictions, None, labels, epoch)
	# GitHub TensorFlow Addons, issue #746, https://github.com/tensorflow/addons/issues/746
	# f1macro.update_state(labels,s_predictions)
	try:
		macro = f1_score(np.squeeze(labels), np.where(np.squeeze(s_predictions) >= 0.5, 1, 0), average='macro',zero_division=0)
	except Exception as ex:
		print(ex)
		macro = 0
		pass
	#
	f1micro.update_state(labels, s_predictions)
	hl.update_state(labels, s_predictions)
	test_loss(loss)
	test_accuracy(labels, s_predictions)
	return macro, s_predictions

def train_model(houses,epochs=1,test_ratio=0.2,drop_probability=0,batch_size=1024,with_teacher=False,unseen=None):
	dset = condense(houses)
	if unseen is not None:
		test_ratio=0	
		train, _trawy  = dset.toSetGenerator(ratio=test_ratio, drop_probability=drop_probability,batch_size=batch_size)
		tset=condense(unseen)
		test, power = tset.toSetGenerator(ratio=test_ratio,batch_size=batch_size)
	else:
		train, test, power = dset.toSetGenerator(ratio=test_ratio, drop_probability=drop_probability,batch_size=batch_size)

	metrics=pd.DataFrame(columns=['Epoch','Train_Loss','Train_Accuracy','Test_Loss','Test_Accuracy','Hamming_Loss','F1_macro','F1_micro','ANE'])
	for epoch in range(epochs):
		train_loss.reset_states()
		train_accuracy.reset_states()
		test_loss.reset_states()
		test_accuracy.reset_states()
		hl.reset_states()
		f1micro.reset_states()
		f1macro = list()

		# ANE
		predictions = list()
		ane = list()

		print("\ttraining...", end='')
		for input_batch, labels in train.as_numpy_iterator():
			train_step(input_batch, labels, epoch,with_teacher)

		print("\ttesting...", end='')
		for input_batch, labels in test.as_numpy_iterator():
			macro,pred =test_step(input_batch, labels, epoch,with_teacher)
			# metrics
			f1macro.append(macro)
			predictions.extend(np.squeeze(pred))
		# f1-macro as average of classes
		f1macro = np.average(f1macro, axis=0)
		## ANE as sum of classes
		predictions = np.array(predictions)
		te_pwr=np.concatenate(list(power))
		e = ANE(predictions, dset.avg_pwr, te_pwr)
		e = np.where(tf.math.is_nan(e), tf.zeros_like(e), e)
		ane.append(tf.reduce_mean(e, axis=0))

		print("\tretrieving metrics:")
		template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
		print(template.format(epoch + 1,
			train_loss.result(),
			train_accuracy.result(),
			test_loss.result(),
			test_accuracy.result()))
		metrics.loc[epoch]=[epoch,train_loss.result().numpy(), train_accuracy.result().numpy(), test_loss.result().numpy(), test_accuracy.result().numpy(),hl.result().numpy(),f1macro,f1micro.result().numpy(),ane[-1].numpy()]
		scores = 'Hamming: {}, f1-Macro: {}, f1-Micro: {}, ANE: {}'
		print(scores.format(hl.result(), f1macro, f1micro.result(), ane[-1].numpy()))
		if es.stop(test_loss.result(), epoch):
			break
	return metrics


#------------------------------------------------#

#------------------------------------------------#
#                     Init                       #
#------------------------------------------------#

s_optimizer = Adam(0.0002)
train_loss = Mean(name='train_loss')
test_loss = Mean(name='test_loss')
train_accuracy = BinaryAccuracy(name='train_accuracy')
test_accuracy = BinaryAccuracy(name='test_accuracy')
hl = HammingLoss(mode='multilabel', threshold=0.5)
f1micro = F1Score(5, average='micro', name='f1_micro', threshold=0.5)
ema = ExponentialMovingAverage(0.99)
es=early(30,80)

#------------------------------------------------#
#------------------------------------------------#
#                     Load                       #
#------------------------------------------------#

# UKDALE
uk_dale = DataSet(os.path.join(PATH_TO_DATASET_DIR,'dale/ukdale.h5'))
appliances = ['kettle', 'microwave', 'dish washer', 'washing machine', 'fridge']
dh1 = house(uk_dale, appliances, 1, '2016-06-01', '2016-08-31')
dh2 = house(uk_dale, appliances, 2, '2013-06-01', '2013-08-05')

## REDD
redd = DataSet(os.path.join(PATH_TO_DATASET_DIR,'redd/redd.h5'))
appliances = ['microwave', 'dish washer', 'washing machine', 'fridge']
rh1 = house(redd, appliances, 1)
rh3 = house(redd, appliances, 3)

#------------------------------------------------#

#------------------------------------------------#
#                     Run                        #
#------------------------------------------------#
