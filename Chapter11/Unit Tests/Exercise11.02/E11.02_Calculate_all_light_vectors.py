########################################################################################
##	Exercise 11.02 – Calculate light vectors of synthetic images of calibration spheres
########################################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Calculate_all_light_vectors import *
import glob
import pathlib


if __name__ == "__main__":
	

	folder_path = r"../Data/Chrome_simulated/"
	sphere_file = r"../Data/Chrome_simulated/sphere_dimensions.txt"
	th = 48
	show = False

	#Run
	L_list = collect_light_vectors(folder_path, sphere_file, th, show)
	print("Lights: \n", L_list)
	print("All vectors Done")
	
