import numpy as np
import cv2
import pathlib
import glob
from scipy.sparse.linalg import lsqr
from form_matrices import *
import matplotlib.pyplot as plt
from matplotlib import cm

#import to register the 3D projection, but is not used directly
from mpl_toolkits.mplot3d import Axes3D

#3d surface plot
def plot_3D(X, Y, Z):		
	fig = plt.figure()
	ax = fig.gca(projection="3d")
	surf = ax.plot_surface(X,Y,Z, cmap= cm.coolwarm,linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()

#Function to calculate depth
def extract_and_save_DepthMap(folder, mask_set):	
	"""
	Input:
	folder: 	Location of the dataset containing subfolder output etc
	mask_set: 	If set to None, a mask is initialized using the dimensions of normal
				Or set it with a mask of shape (normal.shape[0], normals.shape[1])
	Output:
	
	
	"""
	normal_npy_name = str(pathlib.Path(folder) / "outputs" / "normals.npy")
	albedo_name = str(pathlib.Path(folder) / "outputs" / "albedo.png")

	#log names
	depth_name = str(pathlib.Path(folder) / "outputs" / "depth.png")
	depth_npy_name = str(pathlib.Path(folder) / "outputs" / "depth.npy")
	
	albedo = cv2.imread(albedo_name, 0)
	normals = np.load(normal_npy_name)
	if mask_set:
		mask = mask_set
	else:
		mask = np.ones(normals.shape[:2])
	
	print("Normals shape: ", normals.shape)
	print("Mask shape: ", mask.shape)
		
	height = mask.shape[0]
	width = mask.shape[1]
	alpha = mask
	depth_weight = 1.0
	# ~ depth = deepcopy(mask)
	depth = albedo
	
	print("Forming Linear equation coefficients...")
	M, b = matrices_for_linear_equation(alpha, normals, depth_weight, depth, height, width)
	
	print("Solving for Least squares solution...")	
	solution = lsqr(M,b)
	x = solution[0]
	depth = x.reshape(mask.shape)
	
	print("Depth values range from ", depth.min(), " to ", depth.max() )
	np.save(depth_npy_name, depth)
	cv2.imwrite(depth_name, np.clip(np.array(depth), 0,255).astype(np.uint8) )
	
	X, Y = np.meshgrid(np.arange(height), np.arange(width))
	Z = depth.T
	print("\n Use Mouse to view the visualized 3D information from different perspectives.")
	plot_3D(X, Y, Z)
	print("End.")

if __name__ == "__main__":
	
	#Input: Use mask_set by assigning it with any particular custom mask of interest.
	folder = "../Data/Chrome_simulated/"
	#folder = "../Data/real_sphere_based_Data2/"
	#folder = "../Data/pig/"
	
	mask_set = None
	
	#Run
	extract_and_save_DepthMap(folder, mask_set)
	print("Depth estimation Done")
	
