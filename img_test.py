import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import glob
import re
import os
import sys

from sklearn import linear_model

import lane_finding as lane

import lane_double_ransac as dlransac
from lane_double_ransac import DLanesRANSACRegressor

import lane_simple_ransac as slransac
from lane_simple_ransac import SLanesRANSACRegressor

### Camera Calibration
objp1 = np.zeros((6*9,3), np.float32)
objp1[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
objp2 = np.zeros((6*8,3), np.float32)
objp2[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)
objp3 = np.zeros((5*9,3), np.float32)
objp3[:,:2] = np.mgrid[0:9, 0:5].T.reshape(-1,2)
objp4 = np.zeros((4*9,3), np.float32)
objp4[:,:2] = np.mgrid[0:9, 0:4].T.reshape(-1,2)
objp5 = np.zeros((6*7,3), np.float32)
objp5[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)
objp6 = np.zeros((6*5,3), np.float32)
objp6[:,:2] = np.mgrid[0:5, 0:6].T.reshape(-1,2)

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('camera_cal/calibration*.jpg')

fig = plt.figure(figsize=(10, len(images)*1.9))
w_ratios = [1 for n in range(3)]
h_ratios = [1 for n in range(len(images))]
grid = gridspec.GridSpec(len(images), 3, wspace=0.0, hspace=0.0,
			 width_ratios=w_ratios, height_ratios=h_ratios)
i = 0

for idx, fname in enumerate(images):
	img = cv2.imread(fname)
	img2 = np.copy(img)
	img_size = (img.shape[1], img.shape[0])
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
	objp = objp1
	if not ret:
		ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
		objp = objp2
	if not ret:
		ret, corners = cv2.findChessboardCorners(gray, (9,5), None)
		objp = objp3
	if not ret:
		ret, corners = cv2.findChessboardCorners(gray, (9,4), None)
		objp = objp4
	if not ret:
		ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
		objp = objp5
	if not ret:
		ret, corners = cv2.findChessboardCorners(gray, (5,6), None)
		objp = objp6

	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)

		cv2.drawChessboardCorners(img2, (corners.shape[1],corners.shape[0]), corners, ret)
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
		img3 = cv2.undistort(img2, mtx, dist, None, mtx)

		ax = plt.Subplot(fig, grid[i])
		ax.imshow(img)
		if i==0:
			ax.set_title('Original Image')
		ax.set_xticks([])
		ax.set_yticks([])
		fig.add_subplot(ax)
		i += 1
		ax = plt.Subplot(fig, grid[i])
		ax.imshow(img2)
		if i==1:
			ax.set_title('Found Corners')
		ax.set_xticks([])
		ax.set_yticks([])
		fig.add_subplot(ax)
		i += 1
		ax = plt.Subplot(fig, grid[i])
		ax.imshow(img3)
		if i==2:
			ax.set_title('Undistorted Image')
		ax.set_xticks([])
		ax.set_yticks([])
		fig.add_subplot(ax)
		i += 1

	else:
		ax = plt.Subplot(fig, grid[i])
		ax.set_title('Corners Not Found! %s'%(fname))
		ax.set_xticks([])
		ax.set_yticks([])
		fig.add_subplot(ax)
		i += 3

plt.show()
### End Camera Calibration

images = glob.glob('test_images/*.jpg')

fig = plt.figure(figsize=(10, len(images)*3))
w_ratios = [1 for n in range(2)]
h_ratios = [1 for n in range(len(images))]
grid = gridspec.GridSpec(len(images), 2, wspace=0.0, hspace=0.0,
			 width_ratios=w_ratios, height_ratios=h_ratios)
i = 0

"""
for img in [orig, augment]:
"""

def undistort_image(image):
	return cv2.undistort(image, mtx, dist, None, mtx)

for idx, fname in enumerate(images):
	img = cv2.imread(fname)
	undist = undistort_image(img)

	ax = plt.Subplot(fig, grid[i])
	ax.imshow(img)

	if i==0:
		ax.set_title('Original Image')

	ax.set_xticks([])
	ax.set_yticks([])

	fig.add_subplot(ax)

	i += 1

	ax = plt.Subplot(fig, grid[i])
	ax.imshow(undist)

	if i==1:
		ax.set_title('Undistortion Image')

	ax.set_xticks([])
	ax.set_yticks([])

	fig.add_subplot(ax)

	i += 1

plt.show()
