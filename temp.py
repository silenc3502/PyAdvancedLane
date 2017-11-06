import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import sys

import numpy as np
import cv2
from sklearn import linear_model

import lane_finding as lane

import lane_double_ransac as dlransac
from lane_double_ransac import DLanesRANSACRegressor

import lane_simple_ransac as slransac
from lane_simple_ransac import SLanesRANSACRegressor

def plot_steering_hist(steerings, title, num_bins=100):
        plt.hist(steerings, num_bins)
        plt.title(title)
        plt.xlabel('Steering Angles')
        plt.ylabel('Images')
        plt.show()

def plot_dataset_hist(dataset, title, num_bins=100):
        steerings = []
        for item in dataset:
                steerings.append(float(item['steering']))
        plot_steering_hist(steerings, title, num_bins)

def add_shadow(img):
	new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	h,w = new_img.shape[0:2]
	mid = np.random.randint(0,w)
	factor = np.random.uniform(0.6,0.8)
	if np.random.rand() > .5:
		new_img[:,0:mid,0] = (new_img[:,0:mid,0] * factor).astype('uint8')
	else:
		new_img[:,mid:w,0] = (new_img[:,mid:w,0] * factor).astype('uint8')
	new_img = cv2.cvtColor(new_img, cv2.COLOR_YUV2BGR)
	return new_img

def shift_horizon(img):
	h, w, _ = img.shape
	horizon = 2 * h / 5
	v_shift = np.random.randint(-h/8,h/8)
	pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
	pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	return cv2.warpPerspective(img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)

def crop(img, c_lx, c_rx, c_ly, c_ry):
	return img[c_lx:-c_rx, c_ly:-c_ry,]

def brightness(img, value=0):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hsv = hsv.astype('int32')
	hsv[:,:,2] += value
	hsv = np.clip(hsv, 0, 255)
	hsv = hsv.astype('uint8')
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_img(img):
	aug_img = brightness(img, random.randint(-20, 20))
	plt.imshow(aug_img)
	plt.savefig('bright.jpg', format='jpg')
	aug_img = shift_horizon(aug_img)
	plt.imshow(aug_img)
	plt.savefig('horizon_shift.jpg', format='jpg')
	aug_img = add_shadow(aug_img)
	plt.imshow(aug_img)
	plt.savefig('add_shadow.jpg', format='jpg')
	return aug_img

cshape = (9, 6)
path = 'camera_cal/'
mtx, dist = lane.calibration_parameters(path, cshape)

# Test calibration on some image.
filenames = os.listdir(path)
lane.test_calibration(path + filenames[11], cshape, mtx, dist)



old_fpath = "/home/sdr/self_drive/p4/where/SDC-Advanced-Lane-Finding/test_images/test17.jpg"

img = cv2.imread(old_fpath)

img = lane.undistort_image(img, mtx, dist)
wimg = lane.warp_image(img, mtx_perp, flags=cv2.INTER_LINEAR)

plt.imshow(img)
plt.savefig('/home/sdr/self_drive/p4/where/SDC-Advanced-Lane-Finding/debug/undist.jpg', format='jpg')

plt.imshow(wimg)
plt.savefig('/home/sdr/self_drive/p4/where/SDC-Advanced-Lane-Finding/debug/inv_pers.jpg', format='jpg')

# Warped image and masks
wimg = lane.warp_image(img, mtx_perp, flags=cv2.INTER_LANCZOS4)
wmasks = lane.warped_masks(img, mtx_perp, thresholds=[25, 30])
lmask, rmask = lane.default_left_right_masks(img, margin=0.1)

# Masks points.
X1, y1 = lane.masks_to_points(wmasks, lmask, order=2, reverse_x=True, normalise=True, dtype=np.float64)
X2, y2 = lane.masks_to_points(wmasks, rmask, order=2, reverse_x=True, normalise=True, dtype=np.float64)

#print("X1: ")
#print(X1)
#print("y1: ")
#print(y1)
#print("X2: ")
#print(X2)
#print("y2")

# Model validation bounds.
left_right_bounds = np.zeros((1, 3, 2), dtype=X1.dtype)
left_right_bounds[0, 0, 0] = 0.25
left_right_bounds[0, 0, 1] = 0.5
left_right_bounds[0, 1, 1] = 0.2
left_right_bounds[0, 2, 1] = 0.1
valid_bounds = np.zeros((0, 3, 2), dtype=X1.dtype)

# Fit regression.
res_threshold = 0.01
l2_scales = np.array([0.005, 0.005], dtype=X1.dtype)
score_lambdas = np.array([10., 1., 1., 1.], dtype=X1.dtype)
n_prefits = 5000
max_trials = 500000
lanes_ransac = SLanesRANSACRegressor(residual_threshold=res_threshold, 
		n_prefits=n_prefits,
		max_trials=max_trials,
		l2_scales=l2_scales,
		score_lambdas=score_lambdas)
lanes_ransac.fit(X1, y1, X2, y2, left_right_bounds)

# Lane mask.
X_lane, y1_lane, y2_lane = lane.predict_lanes(lanes_ransac, wimg, reversed_x=True, normalised=True)
x_lane = X_lane[:, 1]
wmask_lane = lane.lanes_to_wmask(wimg, x_lane, y1_lane, x_lane, y2_lane)
wimg_lane = cv2.addWeighted(wimg, 0.9, wmask_lane, 0.4, 0.)

# Unwarp everything!
mask_lane = lane.warp_image(wmask_lane, mtx_perp_inv, flags=cv2.INTER_NEAREST)
img_lane = cv2.addWeighted(img, 0.9, mask_lane, 0.4, 0.)

print("img_lane:")
print(img_lane)

# Add curvature and position information.
curv_left = lane.lane_curvature(lane.rescale_coefficients(wimg, lanes_ransac.w1_, perp_scaling))
curv_right = lane.lane_curvature(lane.rescale_coefficients(wimg, lanes_ransac.w2_, perp_scaling))
position = -(lanes_ransac.w1_[0] + lanes_ransac.w2_[0]) / 2. * img.shape[1] * perp_scaling[1]

cv2.putText(img_lane, "Left curvature: %.1f m." % curv_left, (50, 70), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
cv2.putText(img_lane, "Right curvature: %.1f m." % curv_right, (50, 120), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
cv2.putText(img_lane, "Off the center: %.1f m." % position, (50, 170), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)

if debug:
	img_lane = lane.debug_frame(img_lane, wimg, wmasks, lmask, rmask, lanes_ransac)
