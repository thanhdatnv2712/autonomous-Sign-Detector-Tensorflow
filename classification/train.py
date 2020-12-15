import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import glob,os,sys,cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=1, dim=(48,48,3),
                 n_classes=6):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = np.empty((self.batch_size), dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img  = cv2.imread(ID)
            img = cv2.resize(img,(48,48))
            img = img/255.
            
            # Store class
            l = ID.split(os.path.sep)[-2]
            y[i] = int(l)-1
            X.append(img)
#         print(len(X),X[0].shape)
        return np.stack(X,0), keras.utils.to_categorical(y, num_classes=self.n_classes)

if __name__ == "__main__":
    list_IDS = []
    for i in range(1,7):
        path = list(glob.glob(f"./classifier/{i}/*.png")) +  list(glob.glob(f"./classifier/{i}/*.ppm"))
        list_IDS+=path
    np.random.shuffle(list_IDS)
    

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = DataGenerator(list_IDS)


    classifier = keras.Sequential()

    # Step 1 - Convolution
    classifier.add( keras.layers.Conv2D(32, (3, 3), input_shape = (48, 48, 3), activation = 'relu'))

    # Step 2 - Pooling
    classifier.add( keras.layers.MaxPooling2D(pool_size = (2, 2)))
    # Adding a second axpooling
    classifier.add( keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add( keras.layers.MaxPooling2D(pool_size = (2, 2)))
    # Adding a third convolutional layer
    classifier.add( keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add( keras.layers.MaxPooling2D(pool_size= (2, 2)))

    # Step 3 - Flattening
    classifier.add( keras.layers.Flatten())

    # Step 4 - Full connection
    classifier.add( keras.layers.Dense(units = 128, activation = 'relu'))
    classifier.add( keras.layers.Dropout(0.5))
    classifier.add( keras.layers.Dense(units = 6, activation = 'softmax'))
    # Compiling the CNN
    # classifier = keras.models.load_model("my_model.h5")
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    classifier.fit(training_set,
    steps_per_epoch = 4863//32,
    epochs = 20)
    # classifier.save('my_model.h5')
    current_model_saved_name= "./model/best_classification"
    tf.saved_model.save(classifier, current_model_saved_name)
    # tf.keras.models.save_model("sign_classification.h5")