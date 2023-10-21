import numpy as np
# import imageio.v3 as iio
from PIL import Image

input_path = "output.png"

# iio_sem_img = iio.imread("SEMANTIC_IMG_152.png")
# iio_depth_img = iio.imread(input_path)
image = Image.open(input_path)
image_array = np.array(image)
print(image_array.shape)
# print(iio_depth_img)

array_input = "outputnumpy.npy"
arr = np.load(array_input)
print(arr.shape)


