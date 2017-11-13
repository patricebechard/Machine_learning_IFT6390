from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from keras.models import Sequential

class ConvNeuralNetwork1D:
    """1D Convolutional neural network classifier using the Keras library."""
    def __init__(self):
        self.built = False
        self.trained = False
	
    def build(self, input_dim, n_filters, ker_size, n_outputs, pool_size=2):
        """Builds and compiles a 1D convolutional neural networks.
		
        # Arguments
            n_filters: Number of outputs of filters in the convolution. int.
            ker_size: Size of the 1D convolutional window. int.
            pool_size: Size of the max pooling window. int.
            n_outputs: Number of outputs computed by the network. Corresponds to
                the number of classes. int.
        """
        self.model = Sequential((Conv1D(filters=n_filters, kernel_size=ker_size,
										activation='relu', input_shape=(input_dim, 1)),
								 MaxPooling1D(pool_size=pool_size),
								 Flatten(),
								 Dense(units=n_outputs, input_dim=n_filters, activation='sigmoid')))
        self.model.compile(loss='categorical_crossentropy',
						   optimizer='adam',
						   metrics=['accuracy'])
		
        self.built = True
	
    def train(self,
			  x_train, y_train,
			  x_test, y_test,
			  n_epochs,
			  show_acc=True):
        """Trains the 1d convolutional neural network.
		
        # Arguments
            x_train: Training data. ndarray of 1D ndarray.
            y_train: Training labels encoded in onehot. ndarray array of ndarray
			    of size equal to the number of classes.
            x_test: Test data encoded in onehot. ndarray of 1D ndarray.
            y_test: Training labels encoded in onehot. ndarray array of ndarray
			    of size equal to the number of classes.
            n_epochs: int.
            show_acc: Show accuracy plot. bool.
	    
	    # Raises
	        RuntimeError: One must build the model before training it.
        """
        if not self.built:
            raise RuntimeError('You haven\'t build your network yet.')
		
        logs = self.model.fit(x_train, y_train,
							  epochs=n_epochs, batch_size=32,
							  validation_data=(x_test, y_test),
							  verbose=2)
		
        self.trained = True
		
        if show_acc:
            plt.figure(1)
			
            plt.plot(np.arange(0, n_epochs), logs.history['acc'], markersize=0.2, label='training accuracy')
            plt.plot(np.arange(0, n_epochs), logs.history['val_acc'], markersize=0.2, label='test accuracy')
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.xlim(0, n_epochs-1)
            plt.ylim(0.5, 1)
            plt.legend(loc='best')
            plt.savefig('acc_plot.pdf')
		
        pred = self.model.predict(x_test)
		
        print(1 - np.abs(np.argmax(pred, axis=1) - np.argmax(y_test, axis=1)).mean())
	
    def predict(self, x):
        """Computes the model's prediction for some 1D array of data.
	
	    # Arguments
	        x: 1D ndarray.
	    
	    # Raises
	        RuntimeError: One must build the model before training it.
	    """
        if not self.trained:
            raise RuntimeError('You haven\'t trained your network yet.')
		
        return np.argmax(self.model.predict(x), axis=1)