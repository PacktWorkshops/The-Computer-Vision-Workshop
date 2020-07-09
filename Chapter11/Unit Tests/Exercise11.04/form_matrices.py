import numpy as np
from scipy.sparse import csr_matrix


def matrices_for_linear_equation(alpha, normals, depth_weight, depth, height, width):
    """
    Generate M,b to solve for x in Mx = b
    Here the solution will the depth.
    
    Input:
    alpha: 			alpha channel of dimension [height, width]
    normals:		normal image of shape [height, width, 3]
    depth_weight:	Greater the value, lower will be the importance to depth and vice versa.
					A tradeoff parameter between normals and estimated depth. 
					Normals help in smoothing the depth surface and this effect is controlled by this parameter.
					
	depth:			depth will be stored for every pixel.
					Shape is [height, width]
	height, width: 	Dimensions of normals estimated.
    
    Output:
    Constants for equation of type Mx = b
    M:				Coefficient 1 of the linear equation
					It is a very large sparse matrix
	
	b:				Coefficient 2 of the linear equation
	"""

    row_indices = []
    col_indices = []
    data_arr = []
    
    b = []
    if depth_weight == None:
        depth_weight = 1

    """
    Goal is to populate M & b for the purpose generating a linear equation:
    
    M: In matrix M, we populate it only for every non-zero element. 
	b: For every M.shape[0] values, we populate b.
	
	Our aim is to formulate the problem as a linear equation,
	Mx = b, where 
			M is [m, height]
			x is [height, width, 3]
			b is [m, width]
    """

    #Populating row_indices,col_indices,data_arr & b
    c = 0
    for row_i in range(height):
        for col_j in range(width):
            k = row_i * width + col_j
            #alpha check
            if alpha[row_i, col_j] != 0:
				#Ensure input depth is not None.
				#Populate b for depth.
                if depth is not None:
                    b.append(depth_weight * depth[row_i, col_j])
                    row_indices.append(c)
                    col_indices.append(k)
                    data_arr.append(depth_weight)
                    c += 1

                if normals is not None:
                    if col_j + 1 <= width - 1 and alpha[row_i, col_j + 1] != 0:
                        # x-axis normals
                        b.append(normals[row_i, col_j, 0])
                        row_indices.append(c)
                        col_indices.append(k)
                        data_arr.append(-normals[row_i, col_j, 2])
                        row_indices.append(c)
                        col_indices.append(k + 1)
                        data_arr.append(normals[row_i, col_j, 2])
                        c += 1
                    if row_i + 1 <= height - 1 and alpha[row_i + 1, col_j] != 0:
                        # y-axis normals
                        b.append(-normals[row_i, col_j, 1])
                        row_indices.append(c)
                        col_indices.append(k)
                        data_arr.append(-normals[row_i, col_j, 2])
                        row_indices.append(c)
                        col_indices.append(k + width)
                        data_arr.append(normals[row_i, col_j, 2])
                        c += 1
    row = c

    row_indices = np.array(row_indices, dtype=np.int32)
    col_indices = np.array(col_indices, dtype=np.int32)
    data_arr = np.array(data_arr, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # Sparse matrix from the data and indices
    A = csr_matrix((data_arr, (row_indices, col_indices)), shape=(row, width * height))

    return A, b
