import tensorflow as tf
import keras
import cv2
import numpy as np
if __name__ == "__main__":
    model = keras.models.load_model("./my_model.h5")

    path = "./classifier/1/0_00001.png"

    img = cv2.imread(path)
    img = cv2.resize(img,(48,48))
    img = img/255.0
    img = np.reshape(img,[1,48,48,3])
    classes = model.predict(img)
    print(classes.argmax(1))