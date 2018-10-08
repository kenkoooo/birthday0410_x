from PIL import Image
import numpy as np
from tqdm import trange

filenames = [
    "./courier_prime_png/Courier_Prime_0.png",
    "./courier_prime_png/Courier_Prime_1.png",
    "./courier_prime_png/Courier_Prime_2.png",
    "./courier_prime_png/Courier_Prime_3.png",
    "./courier_prime_png/Courier_Prime_4.png",
    "./courier_prime_png/Courier_Prime_5.png",
    "./courier_prime_png/Courier_Prime_6.png",
    "./courier_prime_png/Courier_Prime_7.png",
    "./courier_prime_png/Courier_Prime_8.png",
    "./courier_prime_png/Courier_Prime_9.png",
    "./courier_prime_png/Courier_Prime_cpar.png",
    "./courier_prime_png/Courier_Prime_div.png",
    "./courier_prime_png/Courier_Prime_minus.png",
    "./courier_prime_png/Courier_Prime_opar.png",
    "./courier_prime_png/Courier_Prime_plus.png",
    "./courier_prime_png/Courier_Prime_times.png",
]

np.random.seed(71)

num_files = len(filenames)
num_image_per_image = 1000

array = np.zeros((num_files, num_image_per_image, 65 * 38))
for file_index in trange(num_files):
    filename = filenames[file_index]
    for image_id in trange(num_image_per_image):
        image = Image.open(filename)
        m = np.random.uniform(0.9, 1.0)
        mh = np.random.uniform(0.9, 1.0)
        mw = np.random.uniform(0.9, 1.0)
        r = np.random.uniform(-15, 15)
        sx = np.random.uniform(-0.1, 0.1)
        sy = np.random.uniform(-0.1, 0.1)
        cur_image = image
        cur_image = cur_image.resize(
            (int(cur_image.width*m), int(cur_image.height*m)))
        cur_image = cur_image.resize(
            (int(cur_image.width*mw), cur_image.height))
        cur_image = cur_image.resize(
            (cur_image.width, int(cur_image.height*mh)))
        cur_image = cur_image.rotate(r, fillcolor=255)

        new_image = Image.new('1', (38, 65), color=255)
        for i in range(cur_image.width):
            for j in range(cur_image.height):
                x = int(i + sy * j)
                y = int(sx * i + j)
                if x < 0 or x >= new_image.width or y < 0 or y >= new_image.height:
                    continue
                new_image.putpixel((x, y), cur_image.getpixel((i, j)))
        for i in range(new_image.width):
            for j in range(new_image.height):
                p = np.random.uniform(0, 1)
                if p > 0.05:
                    continue
                value = 0
                if new_image.getpixel((i, j)) == 0:
                    value = 255
                new_image.putpixel((i, j), value)
        array[file_index, image_id] = np.array(new_image).flatten()

np.save("generated_data.npy", array)
