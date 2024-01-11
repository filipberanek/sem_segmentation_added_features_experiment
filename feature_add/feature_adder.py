import os
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
from tqdm import tqdm

# Add canny edges
def add_canny_edges(img_arr:np.array):
    canny = cv2.Canny(np.array(img_arr), 50, 200, None, 3)
    return canny

# Add sobel edges
def sobel_image_norm(image):
    grad_norm = (image * 255 / image.max()).astype(np.uint8)
    return grad_norm

def add_sobel_edges(img_arr:np.array):
    sobelx = cv2.Sobel(src=img_arr, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobelx = sobel_image_norm(sobelx)
    sobely = cv2.Sobel(src=img_arr, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobely = sobel_image_norm(sobely)
    sobelxy = cv2.Sobel(src=img_arr, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    sobelxy = sobel_image_norm(sobelxy)
    sobel_imgs = np.dstack([np.dstack([sobelx, sobely]), sobelxy ])
    return sobel_imgs

def add_sift(img_arr:np.array):
    # Converting image to grayscale
    gray= cv2.cvtColor(img_arr,cv2.COLOR_BGR2GRAY)

    # Applying SIFT detector
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    img_sift=cv2.drawKeypoints(gray ,
                        kp ,
                        img_arr ,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_sift

def add_harris_corner(img_arr:np.array):
    gray = cv2.cvtColor(img_arr,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    harr_cor = img_arr.copy()
    harr_cor[dst>0.01*dst.max()]=[0,0,255]
    return harr_cor

def concatenate_and_save(input_file:pathlib.Path, outpath:pathlib.Path, image_arr:np.array, added_arr:np.array):
    img_arr_moved = np.moveaxis(image_arr, -1,0)
    concat_arr = np.vstack([img_arr_moved,added_arr])
    concat_arr = np.moveaxis(concat_arr, 0,-1)
    np.savez(str(outpath/(input_file.stem + ".npz")), data = concat_arr)

def image_feature_adder(input_path:pathlib.Path, outpath:pathlib.Path, image_size:int):
    canny_output = outpath / (input_path.name+"Added"+"Canny")
    canny_output.mkdir(exist_ok=True, parents=True)
    sobel_output = outpath / (input_path.name+"Added"+"Sobel")
    sobel_output.mkdir(exist_ok=True, parents=True)
    sift_output = outpath / (input_path.name+"Added"+"Sift")
    sift_output.mkdir(exist_ok=True, parents=True)
    harris_output = outpath / (input_path.name+"Added"+"Harris")
    harris_output.mkdir(exist_ok=True, parents=True)
    original_output = outpath / (input_path.name+"Original")
    original_output.mkdir(exist_ok=True, parents=True)
    for file_path in tqdm(list(input_path.rglob("*.png"))):
        img = Image.open(file_path)
        img = img.resize((image_size,image_size))
        img_arr = np.array(img)
        np.savez(str(original_output/(file_path.stem + ".npz")), data = img_arr)
        canny_img = add_canny_edges(img_arr)
        canny_img = np.expand_dims(canny_img, axis=0)
        concatenate_and_save(file_path, canny_output, img_arr, canny_img)
        sobel_img = add_sobel_edges(img_arr)
        sobel_img = np.moveaxis(sobel_img, -1,0)
        concatenate_and_save(file_path, sobel_output, img_arr, sobel_img)
        sift_img = add_sift(img_arr)
        sift_img = np.moveaxis(sift_img, -1,0)
        concatenate_and_save(file_path, sift_output, img_arr, sift_img)
        harris_img = add_harris_corner(img_arr)
        harris_img = np.moveaxis(harris_img, -1,0)
        concatenate_and_save(file_path, harris_output, img_arr, harris_img)


