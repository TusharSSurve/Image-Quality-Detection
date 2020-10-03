import cv2 
import numpy as np 
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
training_folder = os.path.join(dir_path,'CERTH_ImageBlurDataset','TrainingSet')

artificial_blur_path = os.path.join(training_folder,'Artificially-Blurred')
natural_blur_path = os.path.join(training_folder,'Naturally-Blurred')
undistorted_path = os.path.join(training_folder,'Undistorted')

def variance_of_laplacian(image):
    ''' Compute the Laplacian of the image and then return the focus
	measure, which is simply the variance of the Laplacian '''
    return np.var(cv2.Laplacian(image, cv2.CV_64F))

def load_train_images(folders):
    X_train = []
    y_train = []
    for folder in folders:
        if folder == artificial_blur_path or folder==natural_blur_path:
            y_val = 1 # Blurred
        else:
            y_val = 0 # Undistorted 
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                X_train.append(variance_of_laplacian(img))
                y_train.append(y_val)
    xdata_arr = np.asarray(X_train)
    ydata_arr = np.asarray(y_train)
    return np.reshape(xdata_arr,(xdata_arr.shape[0],1)),ydata_arr

train_data_folders = [artificial_blur_path,natural_blur_path,undistorted_path]
X_train,y_train = load_train_images(train_data_folders)

# As it takes a long time to get the variance of laplacian every time we run, it will be a better choice to save the results.
np.save('X_train.npy',X_train)
np.save('y_train.npy',y_train)