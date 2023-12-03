import numpy as np

import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave

import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


train_dir = "dataset"

# Resize images
img_width = 224
img_height = 224
image_size = (img_width, img_height)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    featurewise_center=True,
)

train = train_datagen.flow_from_directory(
    batch_size=128, directory=train_dir, target_size=image_size, class_mode=None
)

x = []
y = []
for img in train[0]:
    try:
        lab = rgb2lab(img)
        x.append(lab[:, :, 0])
        y.append(lab[:, :, 1:] / 128)
    except:
        print("error")

x = np.array(x)
y = np.array(y)

x = x.reshape(x.shape + (1,))
print(x.shape)
print(y.shape)

# Encoder
model = Sequential()
model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        strides=2,
        input_shape=(224, 224, 1),
    )
)
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=2))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same", strides=2))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))

# Decoder
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
model.add(Conv2D(2, (3, 3), activation="tanh", padding="same"))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model.summary()

history = model.fit(
    x=x, y=y, validation_split=0.2, epochs=1000, batch_size=128, verbose=0
)
model.save("portraitColorization2.keras")


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
