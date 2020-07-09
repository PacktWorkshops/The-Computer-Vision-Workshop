import numpy as np
import cv2
import pathlib
import glob


def fetch_dataset(path):
	return Dataset(path)

class Dataset(object):
	def __init__(self, path):
		
		#Data handle
		#self.images & self.light_vectors will be used from every dataset
		im_path = pathlib.Path(path) / "*.png"
		im_names = glob.glob(str(im_path) )
		self.images = []
		for i, name in enumerate(im_names):
			self.images.append( np.float32( cv2.imread(name) ) )
		
		
		lights_path = pathlib.Path(path) / "Estimated_lightvectors.txt"
		self.light_vectors = np.loadtxt(str(lights_path)).T
		
		print("Total number of images read: ", len(self.images))
		print("Light vectors: ", np.array(self.light_vectors), "\n")
		
		#For output
		self.output_path = pathlib.Path(path) / "outputs"
		pathlib.Path(self.output_path).mkdir(exist_ok=True, parents=True)
		
		self.albedo_path = str(self.output_path / "albedo.png")
		self.normals_path = str(self.output_path / "normals.png")
		self.normals_npy = str(self.output_path / "normals.npy")
		
		
		

if __name__ == "__main__":
	
	path = "../Data/turtle/"
	data = fetch_dataset(path)
	
