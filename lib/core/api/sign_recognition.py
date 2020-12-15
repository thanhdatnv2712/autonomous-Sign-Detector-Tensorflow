import tensorflow as tf
import keras
import cv2
import numpy as np

class SignRecognition():
    def __init__(self, model_path):
        self.model = self.get_model()
        self.model.load_weights(model_path)

    def get_model(self):
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
        return classifier

    def predict(self, img):
        img = cv2.resize(img,(48,48)) / 255.0
        img = np.reshape(img,[1,48,48,3])
        classes = model.predict(img)
        return classes.argmax(1)

    def predict_on_batch(self, batch_img, img):
        imgs= []
        for i, pred_i in enumerate(batch_img):
            if pred_i.shape[0] > 0:
                for bbox in pred_i:
                    sign= img[bbox[1] : bbox[3], bbox[0] : bbox[1], ]
                    imgs.append(sign)
        imgs= np.array(imgs)
        classes= model.predict_on_batch(imgs)
        return classes.argmax(1, axis=1)