# Image-Quality-Detection
Deep learning based computer vision project to check whether if image is blurred or not. Using CERTH Image Blur Dataset to achieve 83% accuracy.
CERTH Image Blur Dataset

    E. Mavridaki, V. Mezaris, "No-Reference blur assessment in natural images using Fourier transform and spatial pyramids", Proc. IEEE International Conference on Image Processing (ICIP 2014), Paris, France, October 2014.

The dataset consists of undistorted, naturally-blurred and artificially-blurred images for image quality assessment purposes. Download the dataset from here: http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip

Unzip and load the files into a directory named CERTH_ImageBlurDataset.

--Theory behind variance of laplacian:-
https://stackoverflow.com/questions/48319918/whats-the-theory-behind-computing-variance-of-an-image

--How to install necessary libraries to run in given programs in either Linux or windows:-

pip install tensorflow
pip install scikit-learn
pip install numpy
pip install pandas 
pip install opencv-python

To install opencv you have to install dlib library also. Which can be done by:-
pip install dlib


Here are mainly 3 files:-
Train.py
Test.py
Model.py

Both Train.py & Test.py are used for preprocessing purpose. As both of them take quite a time, I have already excited them and store their results in X_train.npy, y_train.npy, X_test.npy & y_test.npy. 

--Directions to execute program:-
To execute the given program you have to use Model.py.

Command to execute Model.py:-
python Model.py

And, you have to include the dataset for the given problem (Image Quality Detection) as it is.

--Details about given program.
I have created 2 models using 2 different libraries (Tensorflow & Scikit-Learn) in both of them I have used Neural Network
