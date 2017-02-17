#import dependencies--------------------------------------------------------------------------------
import tensorflow as tf
import numpy as numpy
import pandas as pandas
import os
import dicom
import matplotlib.pyplot as plt

#retrieve some data---------------------------------------------------------------------------------
path='G:\Data Science bowl 2017'

#preparing the network------------------------------------------------------------------------------
filter=[3,3,3]

def create_weights(shape):
	return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))

def create_bias(length):
	return tf.Variable(tf.constant(0.1,shape=[length]))