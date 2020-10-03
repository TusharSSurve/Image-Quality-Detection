# import tensorflow as tf 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import os
import cv2 

dir_path = os.path.dirname(os.path.realpath(__file__))
eval_folder = os.path.join(dir_path,'CERTH_ImageBlurDataset','EvaluationSet')
digital_blur_folder = 'DigitalBlurSet'
natural_blur_folder = 'NaturalBlurSet'

digital_blur_path = os.path.join(eval_folder,digital_blur_folder)
natural_blur_path = os.path.join(eval_folder,natural_blur_folder)

natural_blur_file = os.path.join(eval_folder,'NaturalBlurSet.xlsx')
digital_blur_file = os.path.join(eval_folder,'DigitalBlurSet.xlsx')

natural_blur_data = pd.read_excel(natural_blur_file)
digital_blur_data = pd.read_excel(digital_blur_file)

digital_blur_data.rename(columns={'Unnamed: 1':'Blur Label'}, inplace=True)
digital_blur_data.rename(columns={'MyDigital Blur':'Image Name'}, inplace=True)

def variance_of_laplacian(image):
	return np.var(cv2.Laplacian(image, cv2.CV_64F))

def load_test_images(folders):
    X_test = []
    y_test = []
    for folder in folders:
        if folder == natural_blur_path:
            df = natural_blur_data
        else:
            df = digital_blur_data
        for filename in os.listdir(folder):
            if folder == natural_blur_path:
                img_name = os.path.splitext(filename)[0]
            else:
                img_name = filename
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                X_test.append(variance_of_laplacian(img))
                img_idx = 0
                for idx,val in df.iterrows():
                    if(img_name in val['Image Name']):
                        img_idx = idx
                        break
                y_val = df.iloc[img_idx,1]
                if y_val == -1:
                    print('h')
                    y_test.append(0)
                else:
                    y_test.append(y_val)
    xdata_arr = np.asarray(X_test)
    ydata_arr = np.asarray(y_test)
    return np.reshape(xdata_arr,(xdata_arr.shape[0],1)),ydata_arr

test_data_folders = [digital_blur_path,natural_blur_path]
X_test,y_test = load_test_images(test_data_folders)
print(X_test.shape)
print(y_test.shape)

# As it takes a long time to get the variance of laplacian every time we run, it will be a better choice to save the results.
np.save('X_test.npy',X_test)
np.save('y_test.npy',y_test)
