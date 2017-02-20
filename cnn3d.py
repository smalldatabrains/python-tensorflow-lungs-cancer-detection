#import dependencies--------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pandas
import os
import dicom
import matplotlib.pyplot as plt

#retrieve basic information about the dataset-------------------------------------------------------
path='G:\Data Science bowl 2017\stage1'
patients=os.listdir(path)

def information(patients=patients):
	nb_slices=[]
	nb_pixels=[]

	for patient in range(len(patients)):
		dcms=os.listdir(path+'\\'+patients[patient])
		nb_slices.append(len(dcms))
		nb_rows=dicom.read_file(path+'\\'+patients[patient]+'\\'+dcms[0]).Rows
		nb_columns=dicom.read_file(path+'\\'+patients[patient]+'\\'+dcms[0]).Columns
		nb_pixels.append([nb_rows,nb_columns])

	print("We are currently working on ",len(patients)," patients")
	print("Minimum number of slices is ",min(nb_slices))
	print("Maximum number of slices is ",max(nb_slices))
	print("Minimum row and columns pixels qty is ",min(nb_pixels))
	print("Maximum row and columns pixels qty is ",max(nb_pixels))
#It seems that all the files have same qty of columns and rows but range for qty of slices is from 94 to 541. 
#I'll then reduce the quantity of sclices of all the files.

information()

#normalization of the data--------------------------------------------------------------------------
def normalize(dcm):
	pixels=(pixels-np.mean(pixels))/np.stddev(pixels)

# display a slice of lung---------------------------------------------------------------------------
dcms=os.listdir(path+'\\'+patients[0])
def visualize_slice(dcm):
	data=dicom.read_file(path+'\\'+patients[0]+'\\'+dcm) #load data from the first patient
	pixels=data.pixel_array
	plt.imshow(pixels)
	plt.show()

visualize_slice(dcms[0])

#load an entire 3d lung in memory-------------------------------------------------------------------
def load_lung(patient):
	lung=[]
	dcms=os.listdir(path+'\\'+patient)
	for dcm in dcms:
		lung.append(dicom.read_file(path+'\\'+patient+'\\'+dcm))
	lung.sort(key = lambda x: int(x.SliceLocation))
	slice_thickness=np.abs(lung[0].SliceLocation-lung[1].SliceLocation)
	return lung,slice_thickness

lung,slice_thickness=load_lung(patients[0])

#convert to Hounsfield units : kernel from Guido Zuidhof on Kaggle website--------------------------
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

lung_hu=get_pixels_hu(lung)

#display a 3D pic of a lung-------------------------------------------------------------------------
def plot_3d(lung):
	print('do nothing')


plot_3d(lung_hu)

#reducing 3D lungs size for network feeding---------------------------------------------------------
def reduce_3d(patient):
	print('Size reduced to 50x50x50 pixels')

#preparing the network------------------------------------------------------------------------------
def create_weights(shape):
	return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))

def create_bias(length):
	return tf.Variable(tf.constant(0.1,shape=[length]))