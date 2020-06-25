#######################################################################################
##	Exercise 11.01 – Generation of synthetic images – calibration spheres
#######################################################################################


import numpy as np
import math
import cv2
import pathlib

def generate_synthetic_spheres(dim, num, output_folder):
	"""
	Inputs:
	dim: 			Dimension of the image to be generated. A (dim,dim,3) image will be generated.
					Choose image size that is appropriate for your compute capabilities.
	num: 			Number of images that needs to be generated
	output_folder:	The output folder where the generated data will be stored 
	"""
	
		
	output_folder = pathlib.Path(output_folder)
	output_folder.mkdir(exist_ok=True, parents=True)

	images = []
	for _ in range(num):
		images.append(np.zeros((dim, dim, 3), dtype=np.float32))

	np.random.seed(7)
	light_vectors = 2 * (np.random.random((3, num)) - 0.5)
	light_vectors[2, :] += 5
	light_vectors[2, :] = np.abs(light_vectors[2, :])
	light_vectors /= np.linalg.norm(light_vectors, axis=0)[np.newaxis, :]

	for i in range(dim):
		for j in range(dim):
			x = (j - float(dim/2)) / float(dim/2)
			y = -(i - float(dim/2)) / float(dim/2)
			if x**2 + y**2 > 1:
				continue

			z = math.sqrt(1 - x**2 - y**2)

			direction_vector = np.array((x, y, z), dtype=np.float32).reshape((3, 1))

			dots = (light_vectors * direction_vector).sum(axis=0).flatten()

			for image, dot in zip(images, dots):
				image[i, j] = max(0, dot)

	for i, image in enumerate(images):
		image = np.array(image*255, dtype=np.uint8)
		name = 'chrome_sphere_%02d.png'% i
		cv2.imwrite(str(output_folder / name ) , image)
	
	#Save sphere dimensions
	sphere_dims = [int(dim/2), int(dim/2), float(dim/2)]
	np.save(str(output_folder / 'sphere_dimensions.npy'), sphere_dims )
	np.savetxt(str(output_folder / 'sphere_dimensions.txt'), sphere_dims)



if __name__ == "__main__":
	
	dim = 256
	num = 20
	output_folder = "../Data/Chrome_simulated/"
	
	generate_synthetic_spheres(dim, num, output_folder)
	print("Done")
	
