########################################################################################
##	Exercise 11.02 – Calculate light vectors of synthetic images of calibration spheres
########################################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Calculate_light_vectors import calculate_light_vectors
import glob
import pathlib

def collect_light_vectors(folder_path, sphere_file, th, show=False):
	"""
	Input:-
	folder_path : 	Folder that contains png images of the chrome sphere [height, width, 3] lit with light sources at various angles
	sphere_file : 	Path of the file describing the dimensions of the circle/sphere. File contains data in the following format [xc, yc, r]
	th 			:	a tuning threshold (0 to 255) that will extract out the
	show		:	When True, it will display the detection being made. Press enter to continue after every detection.
					When False, the program focuses of detection process alone and saves the Light source direction list in the same folder
	
	Returns:-
	L_list: 				calculated light source directions for all images
	"""
	ppath = pathlib.Path(folder_path) / "*.png"
	ppath = glob.glob(str(ppath))

	sphere_dim = np.loadtxt(sphere_file)
	
	L_list = []
	for i, name in enumerate(sorted(ppath) ):
		im = cv2.imread(name)
		L = calculate_light_vectors(im, sphere_dim, th, show)
		L_list.append(L)

	L_list = np.array(L_list).T
	
	np.save(str(pathlib.Path(folder_path) / "Estimated_lightvectors.npy"), L_list)
	np.savetxt(str(pathlib.Path(folder_path) / "Estimated_lightvectors.txt"), L_list )
	
	return np.array(L_list)

