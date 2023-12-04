import numpy as np
import os
import sys

import matplotlib.pyplot as plt

import skimage
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave

import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def image2lab(image):
    test_img = []
    img = skimage.io.imread(image)
    img = resize(img, (224, 224, 3), anti_aliasing=True)
    lab = rgb2lab(img)
    test_img.append(lab[:, :, 0])

    test_img = np.array(test_img)

    test_img = test_img.reshape(test_img.shape + (1,))
    return test_img


def lab2grayscale(image):
    grayscale = np.zeros((224, 224, 3))
    grayscale[:, :, 0] = image[0][:, :, 0]
    grayscale = resize(grayscale, (800, 600))
    gray_img = lab2rgb(grayscale)
    return gray_img


def color(img):
    model = tf.keras.models.load_model("models/portraitColorization2.keras")

    test_img = image2lab(img)

    output1 = model.predict(test_img)
    output1 = output1 * (128 )

    result = np.zeros((224, 224, 3))
    result[:, :, 0] = test_img[0][:, :, 0]
    result[:, :, 1:] = output1[0]
    result = resize(result, (800, 600))
    color_img = lab2rgb(result)

    return color_img


def display(gray_img, color_img):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()

    ax[0].imshow(gray_img)
    ax[0].set_title("Grayscale Image")

    ax[1].imshow(color_img)
    ax[1].set_title("Colorized Image")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
