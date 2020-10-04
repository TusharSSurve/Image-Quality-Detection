# Importing required libraries
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Loading Training & Testing data from .npy files
X_train, y_train = np.load('X_train.npy'),np.load('y_train.npy')
X_test, y_test = np.load('X_test.npy'),np.load('y_test.npy')

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


# Using Scikit-Learn MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,16),max_iter=500,random_state=42)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print('Accuracy: ',accuracy_score(y_test, predictions)*100.0)


# Using Tensorflow
def get_model():
    model = Sequential([
        Dense(10, input_shape = (1,), activation = 'relu'),
        Dense(20,activation='relu'),
        Dense(1,activation='sigmoid')
    ])

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model
    
tf.random.set_seed(42)
model = get_model()
print(model.summary())

history = model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs=7)

# Loss Graph 
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'],label = "loss")
plt.plot(history.history['val_loss'],label = "val_loss")
plt.legend()
plt.show()
