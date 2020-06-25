############################################################################
##	Exercise 11.03 - Albedo and normal calculation from dataset
############################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pathlib
import glob
from Data_pipeline import *
from tqdm import tqdm


def run_photometric_stereo(light_vectors, images):
	"""
	Input:
	light_vectors: list of light vectors of all images
	images: list of images that needs to be reconstructed
	Returns:
	albedo: albedo map
	normals: normal map
	"""
	light_vectors = np.array(light_vectors)
	light_vectors_transpose = light_vectors.T
	albedo = np.zeros( list(images[0].shape),  dtype=np.float32)
	normals = np.zeros( (*list(images[0].shape)[:2], 3),  dtype=np.float32)
	
	print("Albedo shape- ", albedo.shape)
	print("Normals shape- ", normals.shape)
	
	dp = light_vectors_transpose.dot(light_vectors) 
	t1 = np.linalg.inv(dp)
	for ch in tqdm(range(images[0].shape[2])):
		for row in tqdm(range(images[0].shape[0])):
			for col in range(images[0].shape[1]):
				im = [(images[i][row][col][ch]).T for i in range(len(images))]
				t2 = light_vectors_transpose.dot(im)  # light_transpose*im
				
				VAL = t1.dot(t2)	#t1*t2
				k = np.round(np.linalg.norm(VAL), 5)
				if k < 1e-6:
					k = 0
				else:
					normals[row][col] += VAL / k
				albedo[row][col][ch] = k
	normals = normals/images[0].shape[2]
	
	return albedo, normals
	
#MAIN- fetch data and run PMS
def run_full_pms_pipeline(path, show=False):
	"""
	path: 	folder path that contains n images and Estimated_lightvectors.txt file
			that contains n light vectors
	show: 	When True, the calculated albedo and normals are displayed
			When False, the program proceeds automatically.
	"""
	data = fetch_dataset(path)

	#Light vectors
	light_vectors = data.light_vectors

	#List of images
	images = data.images

	#PMS
	albedo, normals = run_photometric_stereo(light_vectors, images)

	#Save
	# ~ plt.imsave(data.albedo_path, np.array(albedo).astype(np.uint8) )
	# ~ plt.imsave(data.normals_path, np.array(normals*255).astype(np.uint8) )
	cv2.imwrite(data.albedo_path, np.array(albedo).astype(np.uint8) )
	cv2.imwrite(data.normals_path, np.clip(np.array(normals*255), 0,255).astype(np.uint8) )
	np.save(data.normals_npy, normals)
	
	if show == True:			
		fig, ax = plt.subplots(2)
		ax[0].imshow(np.array(albedo).astype(np.uint8))
		ax[1].imshow(np.clip(np.array(normals*255), 0,255).astype(np.uint8))
		plt.show()
	
	return albedo, normals, data


if __name__ == "__main__":
		
	data_path = "../Data/Chrome_simulated/"
	
	albedo, normals, data = run_full_pms_pipeline(data_path)
	print("Done")
	
