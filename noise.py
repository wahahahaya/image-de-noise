import numpy as np
import random
import os
import cv2
import glob

def gauss(image, sigma):
    image_size = np.zeros(image.shape[:2], np.uint8)
    gaussian = cv2.randn(image, 0, sigma)
    return gaussian

def sp(image, amount):
    rows, cols = image.shape
    salt_vs_pepper_ratio = 0.5
    image_sp = image.copy()
    num_salt = np.ceil(amount * image.size * salt_vs_pepper_ratio)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image_sp[coords] = 255
    num_pepper = np.ceil(amount * image.size * (1 - salt_vs_pepper_ratio))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image_sp[coords] = 0
    return image_sp

ground_root = 'data/train/ground'
gauss_root = 'data/train/gauss'
sp_root = 'data/train/sp'
noise_root = 'data/train/noise'
files = os.listdir(ground_root)
for i in range(0, len(files)):
    dir = ground_root + '/' + files[i]
    img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512))
    sp_img = sp(img, random.uniform(0, 1))
    gauss_img = gauss(img, random.randint(0, 50))
    noise = cv2.add(sp_img, gauss_img)
    noise_dir = noise_root + '/' + files[i]
    gauss_dit = gauss_root + '/' + files[i]
    sp_dir = sp_root + '/' + files[i]
    cv2.imwrite(noise_dir, noise)
    cv2.imwrite(gauss_dit, gauss_img)
    cv2.imwrite(sp_dir, sp_img)

