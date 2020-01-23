import numpy as np
#------------------------------
########DEPRECATED#############
#------------------------------

def standardize(x, with_statistics=True):
	m = np.mean(x)
	v = np.std(x)
	n = ((x - m) / v)

	if with_statistics:
		return {'normalized' : n, 'mean' : m, 'variance' : v}
	else:
		return n

def normalize(x):
	mx = np.max(x)
	return x / mx

def standardize_2d(x, with_statistics=True):
	#Standardize each row of 2D arr [X, Y]. Standardize [1, :], [2, :], ...
	m = np.mean(x, axis=1)[:, np.newaxis]
	v = np.std(x, axis=1)[:, np.newaxis]
	n = ((x - m) / v)

	if with_statistics:
		return {'normalized' : n, 'mean' : m, 'variance' : v}
	else:
		return n
