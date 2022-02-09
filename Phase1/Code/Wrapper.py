#!/usr/bin/env python

"""
CMSC733 Spring 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Author(s): 
Saurabh Palande(spalande@umd.edu)
Masters in Robotics Engineering,
University of Maryland, College Park

"""

# Code starts here:
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from sklearn.cluster import KMeans
import sys
# function to generate gaussian filter
def gaussian_filter(filter_size, sigma_x, sigma_y):
  size = int((filter_size-1)/2)
  a = np.asarray([[x**2/sigma_x**2+ y**2/sigma_y**2 for x in range(-size, size+1)] for y in range(-size, size+1)])
  g_filter = (1/(2*np.pi*sigma_x*sigma_y))*np.exp(-a)
  return g_filter

# function to generate DoG filters
def DoG(path):
  s = 2
  o = 16
  sobel_filter = np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
  sigma = np.array([1,4])
  orientations = np.arange(0,360,360/o)
  plt.figure(figsize=(15,2))
  dog_filterbank=[]
  for i in range(0, len(sigma)):
    dog_filter = cv2.filter2D(gaussian_filter(15,sigma[i],sigma[i]),-1, sobel_filter)
    for j in range(0,o):
      b = rotate(dog_filter,orientations[j])
      dog_filterbank.append(b)
      plt.subplot(s,o, j+o*i+1)
      plt.imshow(dog_filterbank[o*i+j],cmap='gray')
      plt.axis('off')
  plt.savefig(path + "/results/filters/DoG.png")
  plt.close()
  return dog_filterbank

def gabor(sigma, theta, Lambda, psi):
    (x, y) = np.meshgrid(np.arange(-24, 24 + 1), np.arange(-24 , 24 + 1))
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * ((x_theta ** 2 + y_theta ** 2) / sigma** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

# functio to generate Gabor filters
def GaborFilter(path):
  s = 5
  o = 8
  theta = np.arange(0,360,360/8)
  sigma = np.array([3,5,7,9, 11])
  plt.figure(figsize=(10,10))
  gf_bank = []
  for i in range(0,len(sigma)):
    for j in range(0, len(theta)):
      g = gabor(sigma[i], theta[j], 5, 0)
      gf_bank.append(g)
      plt.subplot(s, o, j+o*i+1)
      plt.imshow(g, cmap='gray')
      plt.axis('off')
  plt.savefig(path + "/results/filters/Gabor.png")
  plt.close()
  return gf_bank

# function to generate LM filters
def LM_filter_bank(path, scales, name):
  
  orientations = np.arange(0,360, 360/6)
  d_x = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
  d_y = np.array([[1, 2, 1], [0, 0 ,0], [-1, -2, -1]])
  derivative_1 = []
  derivative_2 = []
  log = []
  gaussian = []
  lm_filter  = []
  plt.figure(figsize=(19,7))
  for i in range(3):
    g = gaussian_filter(59, scales[i], 3*scales[i])
    g1 = cv2.filter2D(g, -1, d_x) + cv2.filter2D(g, -1, d_y)
    g2 = cv2.filter2D(g1, -1, d_x) + cv2.filter2D(g1, -1, d_y)
    for j in range(len(orientations)):
      d1 = rotate(g1, orientations[j])
      derivative_1.append(d1)
      d2 = rotate(g2, orientations[j])
      derivative_2.append(d2)

  for i in range(3):
    for j in range(len(orientations)):
      lm_filter.append(derivative_1[j+len(orientations)*i])
    for k in range(len(orientations)):
      lm_filter.append(derivative_2[k+len(orientations)*i])

  for i in range(len(scales)):
    g = gaussian_filter(59, scales[i], scales[i])
    l = cv2.filter2D(g, -1, np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]]))
    log.append(l)
    lm_filter.append(l)

  for i in range(len(scales)):
    g = gaussian_filter(59, 3*scales[i], 3*scales[i])
    l = cv2.filter2D(g, -1, np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]]))
    log.append(l)
    lm_filter.append(l)

  for i in range(len(scales)):
    g = gaussian_filter(59, scales[i], scales[i])
    gaussian.append(g)
    lm_filter.append(g)

  for i in range(len(lm_filter)):
    plt.subplot(4,12, i+1)
    plt.imshow(lm_filter[i], cmap='gray')
    plt.axis('off')
  plt.savefig(path + "/results/filters/"+name+".png")
  plt.close()
  return lm_filter

def texton_map(img,filter_bank):

  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  texture_properties = []
  for i in range(len(filter_bank)):
    filter_img = cv2.filter2D(gray_img,-1, filter_bank[i])
    texture_properties.append(filter_img)
  
  texture_properties = np.array(texture_properties)
  x,y,z = texture_properties.shape
  t = texture_properties.reshape((x, y*z))
  t = t.transpose()
  kmeans = KMeans(n_clusters=64, n_init = 2)
  kmeans.fit(t)
  labels = kmeans.labels_
  g = labels.reshape((y,z))
  return g

def b_map(img):
  g= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  n = g.reshape((g.size),1)
  kmeans = KMeans(n_clusters=16, n_init=5)
  kmeans.fit(n)
  labels = kmeans.labels_
  brightness_map = labels.reshape(g.shape)
  return brightness_map

def c_map(img):
  q= img.reshape((img.shape[0]*img.shape[1]),3)
  kmeans = KMeans(n_clusters=16, n_init=5)
  kmeans.fit(q)
  labels = kmeans.labels_
  color_map = labels.reshape((img.shape[0],img.shape[1]))
  return color_map

def half_disk(radius):
	size = 2*radius + 1
	c= radius
	half_disk = np.zeros([size, size])
	for i in range(radius):
		for j in range(size):
			distance = np.square(i-c) + np.square(j-c)
			if distance <= np.square(radius):
				half_disk[i,j] = 1

	
	return half_disk

def hd_filter(path):
  hd_filter_bank=[]
  radius=np.array([5,10, 20, 30, 40])
  orientations = np.arange(0,360, 360/8)
  plt.figure(figsize=(10,10))
  for i in range(len(radius)):
    hd_filter = half_disk(radius[i])
    for j in range(int(len(orientations)/2)):
      h = rotate(hd_filter, orientations[j])
      h[h<=0.5] = 0
      h[h>0.5] = 1
      hd_filter_bank.append(h)
      h1 = rotate(hd_filter, orientations[j]+180)
      h1[h1<=0.5] = 0
      h1[h1>0.5] = 1
      hd_filter_bank.append(h1)

  for i in range(len(hd_filter_bank)):
    plt.subplot(6,8, i+1)
    plt.imshow(hd_filter_bank[i],cmap='gray')
    plt.axis('off')
  plt.savefig(path + "/results/filters/half_disc.png")
  plt.close()
  return hd_filter_bank

def chi_square(t, hd_mask, bin_size):
  chi_sqr_dist = []
  for i in range(0,len(hd_mask),2):
    tmp = np.zeros(t.shape)
    chi_distance = np.zeros(t.shape)
    for j in range(bin_size):
      tmp[t==j] = 1
      g = cv2.filter2D(tmp, -1, hd_mask[i])
      h = cv2.filter2D(tmp,-1, hd_mask[i+1])
      chi_distance += ((g-h)**2/(g+h+0.00001))

    chi_sqr_dist.append(chi_distance/2)
  return chi_sqr_dist

def main(): 

  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  # sys.path.append(BASE_DIR)
  data_path = os.path.dirname(os.getcwd())
  images_path = os.path.join(data_path, 'BSDS500/Images')
  sobel_path = os.path.join(data_path, 'BSDS500/SobelBaseline')
  canny_path = os.path.join(data_path, 'BSDS500/CannyBaseline')
  d = DoG(BASE_DIR)
  print("Generated DoG filter")
  g = GaborFilter(BASE_DIR)
  print("Generated Gabor filter")
  scales_lms= np.array([1, np.sqrt(2), 2, 2*np.sqrt(2)])
  lms = LM_filter_bank(BASE_DIR, scales_lms, "lms")
  print("Generated LMS filter")
  scales_lml= np.array([np.sqrt(2), 2, 2*np.sqrt(2), 4])
  lml = LM_filter_bank(BASE_DIR, scales_lml, "lml")
  print("Generated LML filter")
  h = hd_filter(BASE_DIR)
  print("Generated half disc filter")
  filter_bank= d + lms+lml+ g
  # Loading the images 
  image_files = os.listdir(images_path)
  for image in image_files:
    image_number, image_type = image.split('.')
    print("Implementing pb-lite on image ", image_number)
    img_path = os.path.join(images_path, image)
    img = cv2.imread(img_path)

    t = texton_map(img,filter_bank)
    plt.imsave(BASE_DIR + "/results/texton/"+ image, t)
    b = b_map(img)
    plt.imsave(BASE_DIR + "/results/b_map/"+ image, b)
    c = c_map(img)
    plt.imsave(BASE_DIR + "/results/c_map/"+ image, c)

    tg = chi_square(t, h, 64)
    tg = np.array(tg)
    tg = np.mean(tg, axis=0)
    plt.imsave(BASE_DIR + "/results/texture_gradient/"+ image, tg)
    bg = chi_square(b, h, 16)
    bg = np.array(bg)
    bg = np.mean(bg, axis=0)
    plt.imsave(BASE_DIR + "/results/brightness_gradient/"+ image, bg)
    cg = chi_square(c, h, 16)
    cg = np.array(cg)
    cg = np.mean(cg,axis=0)
    plt.imsave(BASE_DIR + "/results/color_gradient/"+ image, cg)
    canny = cv2.imread(canny_path + "/" + str(image_number) + ".png")
    canny_baseline = cv2.cvtColor(canny,cv2.COLOR_BGR2GRAY)
    sobel = cv2.imread((sobel_path + "/" + str(image_number) + ".png"))
    sobel_baseline = cv2.cvtColor(sobel, cv2.COLOR_BGR2GRAY)
    p1 = (tg+bg+cg)/3
    p2 = 0.5*canny_baseline + 0.5*sobel_baseline
    pb_lite  = np.multiply(p1,p2)
    plt.figure()
    plt.imshow(pb_lite, cmap = 'gray')
    plt.imsave(BASE_DIR + "/results/pb_lite/"+ image, pb_lite, cmap='gray')
    # plt.show()
    
if __name__ == '__main__':
    main()