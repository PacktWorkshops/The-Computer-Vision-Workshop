###########################################################################
##	Activity 11.02- Finding albedo and normals in a real dataset.
###########################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pathlib
import glob
from Data_pipeline import *
from tqdm import tqdm
from photometric_stereo import *

if __name__ == "__main__":
		
	data_path = "../Data/real_sphere_based_Data1/"
	
	albedo, normals, data = run_full_pms_pipeline(data_path)
	print("Done")
	
