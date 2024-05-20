#matrix manipulation - convert color image to grayscale
#do not make changes in the function names

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def to_grayscale(image):

    l, m, n = 0.2989, 0.5870, 0.1140,
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    grayscale = l*r + m*g + n*b

    return grayscale


def main():

    # use the already given image only for submission
    image_path = 'colorImage.jpg'

    # Load the colored image here
    image = Image.open(image_path)
    image_matrix = np.array(image)

    # save the grayscale image as grayscale_image in same folder
    grayscale_matrix = to_grayscale(image_matrix)
    grayscale_image = Image.fromarray(grayscale_matrix).convert('RGB')
    grayscale_image.save('./grayscale_image.jpg')

if __name__ == "__main__":
    main()
