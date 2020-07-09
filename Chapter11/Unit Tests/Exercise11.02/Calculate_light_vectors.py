import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

#Orthographic camera model is assumed. 
# The origin lies on the sphere/circle used
# Camera is at (0,0,1)

def calculate_light_vectors(im, sphere_dim, th=200, show=False):
	"""
	Input:-
	im : 			the image of the chrome sphere [height, width, 3] lit with a light source at an angle
	sphere_dim : 	a vector [xc, yc, r] describing the dimensions of the circle
	th : 			a tuning threshold (0 to 255) that will extract out the
	
	Returns:-
	l: 				calculated light source direction	
	"""
	
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	xc = sphere_dim[0]
	yc = sphere_dim[1]
	radius = sphere_dim[2]
	print("Read sphere params: ", xc,yc,radius)
	
	#Extract the lit area
	x_indices = []
	y_indices = []
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			val = im[i, j]
			if val > th:
				x_indices.append(i)
				y_indices.append(j)
	
	x_med = np.median(x_indices)
	y_med = np.median(y_indices)
	print("Detected point is ({},{})".format(x_med, y_med) )
	
	im2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
	im2 = cv2.circle(im2, (int(y_med), int(x_med)), radius=3, color=(0,0,255), thickness=5)
	
	if show == True:
		print("Press Enter to Proceed....")
		cv2.imshow("Detected point", im2)
		cv2.waitKey(0)
	
	#Normal to lit point on sphere
	n = [x_med - xc, y_med - yc]
	n = n/radius
	n3 = math.sqrt(1 - (n[0]*n[0]) - (n[1]*n[1]) )
	
	#Normal vector to the lit point is N
	N = np.array([*n, n3])
	print("\nNormal vector to the lit point is ", N)
	
	#Find light source direction when the viewing angle is (0, 0, 1)
	L = 2 * N[2] * N
	V = [0, 0, 1]
	L = np.subtract(L, V)
	print("Light source direction is ", L)
	
	return L
	

	
if __name__ == "__main__":
	
	path = r"../Data/Chrome_simulated/chrome_sphere_00.png"
	sphere_file = r"../Data/Chrome_simulated/sphere_dimensions.txt"
	th = 45

	im = cv2.imread(path)
	sphere_dim = np.loadtxt(sphere_file)
	
	l = calculate_light_vectors(im, sphere_dim, th, True)
	print("Single vector Done")
	

