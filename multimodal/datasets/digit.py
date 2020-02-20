from sklearn import datasets
import numpy as np
import PIL
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from multimodal.datasets.base import load_dict, save_dict
from multimodal.tests.data.get_dataset_path import get_dataset_path
from multimodal.datasets.data_sample import MultiModalArray
from multimodal.kernels.mvml import MVML
#Load the digits dataset
digits = datasets.load_digits()

#Display the first digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
colors = digits.data
gradiant = np.gradient(digits.images, axis=[1,2])
print(gradiant[0].shape)
gradiant0 = gradiant[0].reshape(colors.shape[0], colors.shape[1])
gradiant1 = gradiant[1].reshape(colors.shape[0], colors.shape[1])
for ind in range(digits.images.shape[0]):
    ima0 = digits.images[ind, :,:]
    ima1 = gradiant[0][ind, :,:]
    ima2 = gradiant[1][ind, :,:]
    ama_pil0 = PIL.Image.fromarray(ima0, mode=None)
    ama_pil1 = PIL.Image.fromarray(ima1, mode=None)
    ama_pil2 = PIL.Image.fromarray(ima2, mode=None)
    histo_color = np.asarray(ama_pil0.histogram())
    histo_gradiant0 = np.asarray(ama_pil1.histogram())
    histo_gradiant1 = np.asarray(ama_pil2.histogram())
    if ind==0:
        list_histogram_color = histo_color
        list_histogram_gradiant0 = histo_gradiant0
        list_histogram_gradiant1 = histo_gradiant1
    else:
        list_histogram_color = np.vstack((list_histogram_color, histo_color))
        list_histogram_gradiant0 = np.vstack((list_histogram_gradiant0, histo_gradiant0))
        list_histogram_gradiant1 = np.vstack((list_histogram_gradiant1, histo_gradiant1))

dict_digit = {0: list_histogram_color, 1: list_histogram_gradiant0, 2: list_histogram_gradiant1}


print(list_histogram_color.shape)
print(list_histogram_gradiant0.shape)
print(list_histogram_gradiant1.shape)
file = get_dataset_path("digit_histogram.npy")
save_dict(dict_digit, file)

d2 = load_dict(file)

figure = plt.figure(figsize=(27, 9))
ax = plt.subplot(2,1,1)

ax.scatter(list_histogram_color[:,3], list_histogram_color[:,4], c=digits.target, edgecolors='k')
ax = plt.subplot(2,1,2)
ax.scatter(list_histogram_color[:,0], list_histogram_color[:,1], c=digits.target, edgecolors='k')
plt.show()

mvml = MVML(lmbda=0.1, eta=1, nystrom_param=0.2)
mvml.fit(d2)
