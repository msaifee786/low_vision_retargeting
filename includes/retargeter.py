###############################################################################
# retargeter.py
#
# Description:
#	Class definition for retargeter. This object takes in a frame and will do
#	energy map calculations and then perform the retargeting based on some
#	algorithm.
#	Major assumptions in this file are:
#		- Leverages OpenCV
#		- All inputted frames are Numpy arrays
#
#	To Do:
#
###############################################################################

# Import necessary packages
import cv2
import sys
import dlib


# Import packages for image processings:
from skimage import transform
from skimage import filters

# Import some helper packages:
import argparse
import time
import math
import numpy as np
from collections import deque

class Retargeter:

	# Initialization method:
	def __init__(self, frame_size_x, frame_size_y, num_seams_x, num_seams_y):
	
		# How many seams should be removed with retargeting:
		self.my_seam_num_v = num_seams_x;
		self.my_seam_num_h = num_seams_y;
		
		# Initialize our energy map:
		self.my_current_energy_map = np.zeros((frame_size_y, frame_size_x));
		self.my_previous_energy_map = np.zeros((frame_size_y, frame_size_x));


	# Instance Method - pass in frame (current and previous) and rectangles specifying
	# ROIs, and calculates the energy map
	# It also saves current/previous energy maps
	def generate_energy_map(self, previous_frame_gray, current_frame_gray, roi_rects):
	
		# Get energy map based on grayscale:
		energy_map = Retargeter.calculate_energy_map(previous_frame_gray, current_frame_gray, roi_rects);

		# Advance along the energy maps:
		self.my_previous_energy_map = self.my_current_energy_map;
		self.my_current_energy_map = energy_map

		return energy_map;
		
	# Instance Method - Retargets a frame based on calculated energy map
	# NOTE - ASSUMES calculate_energy_map HAS BEEN CALLED ON THE FRAME
	def retarget_frame(self, frame):
	
		# To Do: average the previous and current energy maps before using; this may help
		# reduce jitter
		energy_map = self.my_current_energy_map;
	
		# Lastly, perform the retargeting:
		retargeted_frame = Retargeter.do_retargeting_algorithm(frame, energy_map, self.my_seam_num_v, self.my_seam_num_h);
		
		return retargeted_frame;
		

		
	# Class method for retargeting
	# Does vertical seams first, then horizontal
	@staticmethod
	def do_retargeting_algorithm(frame, energy_map, num_seams_v, num_seams_h):
	
		# For now, just do seam carving
		# To Do: get a better (less jittery, faster) retargeting algorithm. Consider
		# implementing it in C via a python-C API
		carved = transform.seam_carve(frame, energy_map, "vertical", num_seams_v);
		#carved = transform.seam_carve(carved, energy_map, "horizontal", num_seams_h)

		return carved;


	# calculate_energy_map
	# Given previous and current frames (and current energy map), finds "energy" in frame
	# Notably, assumes frames are Numpy arrays. See below for implementation
	@staticmethod
	def calculate_energy_map(previous_frame, current_frame, roi_rects):

		# TODO:
		# Fix weighting between temporal and spatial energy
	
	
		# Overall steps:
		# 1. Calculate spatial energy (via simple filtering)
		# 2. Increase energy of ROI rectangles (to preserve them on carving)
		# 3. Calculate temporal frequency (via helper function)
		# 4. Add spatial + temporal frequency for final value
	
		# Get spatial energy via edge detection filter
		spatial_freq_energy = Retargeter.calculate_spatial_energy(current_frame);

		# Now, increase energy of ROIs:
		for (x, y, w, h) in roi_rects:
			spatial_freq_energy[y:(y+h), x:(x+w)] = 255;

		# Now, deal with temporal frequency energy:
		temp_freq_energy = Retargeter.calculate_temporal_energy(current_frame, previous_frame);

		# Add temporal frequency to spatial to get our full energy:
		mag = spatial_freq_energy + temp_freq_energy;

		return mag;

	# calculate_spatial_energy
	# Given a (grayscale) frame, calculates spatial energy as some calculation of spatial
	# frequency/edge detection/derivatives
	@staticmethod
	def calculate_spatial_energy(frame):
	
		# First blur the image, to filter out some noise:
		blurred = cv2.GaussianBlur(frame, (3,3), 0, 0);
		
		# Use Sobel filter:
		return Retargeter.sobel_filter(blurred);


	# calculate_temporal_energy
	# Given current and previous frame, calculates temporal energy
	# First we blur, then take absolute difference between frames and threshold based on a 
	# cutoff value. Then we dilate those areas to make continuous contours
	@staticmethod
	def calculate_temporal_energy(current_frame, previous_frame):

		# First, blur both to help reduce noise
		blur_window_size = 21;
	
		current_frame = cv2.GaussianBlur(current_frame, (blur_window_size, blur_window_size), 0);
		previous_frame = cv2.GaussianBlur(previous_frame, (blur_window_size, blur_window_size), 0);
	
		# Use get absolute difference between the two frames
		frame_difference = cv2.absdiff(current_frame, previous_frame);
	
		# Now, filter based on threshold --> 0 or ceiling value

		# Define difference threshold as % change
		threshold_cutoff_value = int(256*0.03);

		# Do thresholding. Note that it returns a tuple (retval, array) so we select for the array
		frame_difference = cv2.threshold(frame_difference, threshold_cutoff_value, 255, cv2.THRESH_BINARY)[1]

		# Dilate areas of potential movement to fill in any holes --> make the areas continuous
		frame_difference = cv2.dilate(frame_difference, None, iterations=3);
	
		return frame_difference;

	# Helper function -- implements canny filter; for use in calculating spatial energy
	@staticmethod
	def canny_filter(frame):
		# Canny filter sigma coefficient (determines how narrow/wide to threshold)
		canny_sigma=0.33

		# compute the median of the single channel pixel intensities
		v = np.median(frame)
 
		# apply automatic Canny edge detection using the computed median
		lower = int(max(0, (1.0 - canny_sigma) * v))
		upper = int(min(255, (1.0 + canny_sigma) * v))
		edged = cv2.Canny(frame, 100, 200)

		# Lastly, dilate the lines to bold them: 
		edged = cv2.dilate(edged, None, iterations=1);
		
		return edged;
		
	# Helper function -- implements 2d sobel filter; for use in calculating spatial energy
	@staticmethod
	def sobel_filter(frame):
	
		# Calculate the x and y Sobel filter. Do so with signed 64 bit values:
		sobelx64 = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
		sobely64 = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)

		# Then, convert to unsigned 8 bit values by taking absolute value:
		sobelx = np.absolute(sobelx64)/2
		sobely = np.absolute(sobely64)/2
	
		# Lastly, combine them to get full derivative value at each point:
		sobel_total = np.uint8(np.add(sobelx, sobely));
			
		return sobel_total;