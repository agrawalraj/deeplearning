
import numpy as np 
import h5py 

# From comma.ai github 
def concatenate(camera_names, time_len):
  logs_names = [x.replace('camera', 'log') for x in camera_names]

  angle = []  # steering angle of the car
  speed = []  # steering angle of the car
  hdf5_camera = []  # the camera hdf5 files need to continue open
  c5x = []
  filters = []
  lastidx = 0

  for cword, tword in zip(camera_names, logs_names):
    try:
      with h5py.File(tword, "r") as t5:
        c5 = h5py.File(cword, "r")
        hdf5_camera.append(c5)
        x = c5["X"]
        c5x.append((lastidx, lastidx+x.shape[0], x))

        speed_value = t5["speed"][:]
        steering_angle = t5["steering_angle"][:]
        idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment
        angle.append(steering_angle[idxs])
        speed.append(speed_value[idxs])

        goods = np.abs(angle[-1]) <= 200

        filters.append(np.argwhere(goods)[time_len-1:] + (lastidx+time_len-1))
        lastidx += goods.shape[0]
        # check for mismatched length bug
        print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0]))
        if x.shape[0] != angle[-1].shape[0] or x.shape[0] != goods.shape[0]:
          raise Exception("bad shape")

    except IOError:
      import traceback
      traceback.print_exc()
      print "failed to open", tword

  angle = np.concatenate(angle, axis=0)
  speed = np.concatenate(speed, axis=0)
  filters = np.concatenate(filters, axis=0).ravel()
  print "training on %d/%d examples" % (filters.shape[0], angle.shape[0])
  return c5x, angle, speed, filters, hdf5_camera

def get_array(hdf5_file):
    data = hdf5_file['X'][:]
    hdf5_file.close()
    return data 

def load_data_label(camera_path):
	c5x, angle, speed, filters, hdf5_camera = concatenate([camera_path])
	data = get_array(hdf5_camera[0])
	return (data, angle, speed)

# Depricated - runs out of memory...
def get_merged_images(num_samps, hdf5_camera):
	X = None 
	curr = 0 
	for i, h5_file in enumerate(hdf5_camera):
		data = h5_file['X'][:]
		h5_file.close()
		if i == 0:
			n, n_channels, w, h = data.shape
			X = np.zeros(num_samps, n_channels, w, h)
			X[:n, :, :, :] = data
			curr += n  
		else:
			n = data.shape[0]
			X[curr:(curr + n), :, :, :] = data
	return X 
