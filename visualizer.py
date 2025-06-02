#import dependencies--------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pandas
import os
import dicom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
	plt.hist(nb_slices)
	plt.show()
information()

#normalization of the data--------------------------------------------------------------------------
def normalize(dcm):
	pixels=(pixels-np.mean(pixels))/np.stddev(pixels)

#plot a slice of lung-------------------------------------------------------------------------------
dcms=os.listdir(path+'\\'+patients[0])
def visualize_slice(dcm):
	data=dicom.read_file(path+'\\'+patients[0]+'\\'+dcm) #load data from the first patient
	pixels=data.pixel_array
	plt.imshow(pixels)
	plt.show()

visualize_slice(dcms[0])

#plot a (z,y) or (z,x) oriented slice---------------------------------------------------------------
def visualize_slice2(full_lung,fixed='x',position=256):
	if fixed =='x':
		plt.imshow(full_lung[position,:,:])
		plt.show()
	elif fixed=='y':
		plt.imshow(full_lung[:,position,:])
		plt.show()

#load an entire 3d lung in memory-------------------------------------------------------------------
def load_lung(patient):
	lung=[]
	dcms=os.listdir(path+'\\'+patient)
	for dcm in dcms:
		lung.append(dicom.read_file(path+'\\'+patient+'\\'+dcm))
	lung.sort(key = lambda x: int(x.SliceLocation))
	slice_thickness=np.abs(lung[0].SliceLocation-lung[1].SliceLocation)
	pixel_spacing=lung[0].PixelSpacing
	return lung,slice_thickness,pixel_spacing

lung,slice_thickness,pixel_spacing=load_lung(patients[0])

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

#plot full 3d version of a lung---------------------------------------------------------------------
def plot_3d(lung):
	x=[x*pixel_spacing for x in range(0,512)]
	y=x
	z=[z*slice_thickness for z in range(0,len(lung))]
	pixel=[]
	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')
	for slice in range(1,len(lung)):
		pixel.append(lung[slice].pixel_array)
	full_lung=np.stack((pixel),axis=2)
	print(full_lung.shape) #x,y,z dimensions


#reducing 3D lungs size for network feeding---------------------------------------------------------
def reduce_size(full_lung):
	target=50
	full_lung.resize((target,target,target))
	return full_lung


#preparing the network------------------------------------------------------------------------------
def create_weights(shape):
	return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))

def create_bias(length):
	return tf.Variable(tf.constant(0.1,shape=[length]))
