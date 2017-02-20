#import dependencies--------------------------------------------------------------------------------
import tensorflow as tf
import numpy as numpy
import pandas as pandas
import os
import dicom
import matplotlib.pyplot as plt

#retrieve basic information about the dataset-------------------------------------------------------
path='G:\Data Science bowl 2017\stage1'
patients=os.listdir(path)

nb_slices=[]
nb_pixels=[]

for patient in range(len(patients)):
	dcms=os.listdir(path+'\\'+patients[patient])
	nb_slices.append(len(dcms))
	nb_rows=dicom.read_file(path+'\\'+patients[patient]+'\\'+dcms[0]).Rows
	nb_columns=dicom.read_file(path+'\\'+patients[patient]+'\\'+dcms[0]).Columns
	nb_pixels.append([nb_rows,nb_columns])

print("Minimum number of slices is ",min(nb_slices))
print("Maximum number of slices is ",max(nb_slices))
print("Minimum row and columns pixels qty is ",min(nb_pixels))
print("Maximum row and columns pixels qty is ",max(nb_pixels))
#It seems that all the files have same qty of columns and rows but range for qty of slices is from 94 to 541. 
#I'll then reduce the quantity of sclices of all the files.



#normalization of the data--------------------------------------------------------------------------
def normalize(dcm):
	pixels=(pixels-np.mean(pixels))/np.stddev(pixels)

# display a slice of lung----------------------------------------------------------------------------
dcms=os.listdir(path+'\\'+patients[0])
def visualize_slice(dcm):
	data=dicom.read_file(path+'\\'+patients[0]+'\\'+dcm) #load data from the first patient
	pixels=data.pixel_array
	plt.imshow(pixels)
	plt.show()
visualize_slice(dcms[0])


#reducing 3D lungs size for network feeding---------------------------------------------------------
def reduce_3d(patient):
	print('Size reduced to 30x30x30 pixels')

#preparing the network------------------------------------------------------------------------------
def create_weights(shape):
	return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))

def create_bias(length):
	return tf.Variable(tf.constant(0.1,shape=[length]))