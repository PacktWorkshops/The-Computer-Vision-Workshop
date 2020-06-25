###########################################################################
##	Activity 11.03 – Finding Depth Map in real dataset
###########################################################################

import numpy as np
import cv2
import pathlib
import glob
from scipy.sparse.linalg import lsqr
from form_matrices import *
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D
from estimate_depth_map import *
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
	
	#Input: Use mask_set by assigning it with any particular custom mask of interest.
	folder = "../Data/real_sphere_based_Data1/"	
	mask_set = None
	
	#Run
	extract_and_save_DepthMap(folder, mask_set)
	print("Depth estimation Done")
	
