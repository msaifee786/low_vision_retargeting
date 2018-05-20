###############################################################################
# image_funcs.py
#
# Description:
#	Helper functions for processing images. Major assumptions in this file are:
#		- Leverages OpenCV
#		- All inputted frames are Numpy arrays
#
#	Note that this is not a class definition; only helper functions
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
import math
import numpy as np
from collections import deque

# magnify_areas_in_frame
# Given a frame, and an array of quadruples (x,y,w,h) specifying rectangles, magnifies the
# area specified by the rectangles based on a scaling factor
# Returns a tuple (new_frame, enlarged_rects)
# NO ERROR CHECKING DONE TO MAKE SURE THAT SCALED REGION WILL FIT IN FRAME
def magnify_areas_in_frame(frame, array_of_rects):

	# Scaling factor
	scale_factor = 2.0;

	# Get current frame, and copy it because scaling one rectangle up may cover up other
	# regions
	output_frame = np.copy(frame);
	
	# Prepare array of enlarged rects for output:
	enlarged_rects = [];

	# For each region:
	for (x, y, w, h) in array_of_rects:
			
 		# Grab the area of interest
		rect = frame[y:y+h, x:x+w];
 		
		#Scale area based on factor
		scaled_rect = cv2.resize(rect, (int(w*scale_factor), int(h*scale_factor)));

		# We know we have a face here, likely in the context of a head.
		# Below is some sample code (currently commented out) to attempt to segment out
		# the face from background
		
		# First, get the edge map:
		# canny_sigma=0.2

		# compute the median of the single channel pixel intensities
		# v = np.median(frame)
		# lower = int(max(0, (1.0 - canny_sigma) * v))
		# upper = int(min(255, (1.0 + canny_sigma) * v))

		#rect_edges = cv2.Canny(scaled_rect, lower, upper)

		# Apply automatic Canny edge detection using the computed median

		# cv2.imshow("canny_edges", rect_edges);
		# cv2.waitKey(0);

		# Then, get the contour list:
		# (contours, _, _) = cv2.findContours(rect_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);

		# We assume the largest contour is the face
		# face_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0];

		# Now make the mask for selecting out face only:
		# mask = np.zeros(rect_edges.shape, np.uint8);

		# cv2.drawContours(mask, face_contour, -1, (255), 1);

		# And now select for face only:
		# rect_edges = cv2.bitwise_and(rect_edges, mask);


		# End sample code for segmenting out face =============================


		# Now, need to paste the scaled region on to original
		# We do this by pasting center of scaled region on center of original rectangle 
		
		# First find center point of the rect:
		center_x = x + int(w/2);
		center_y = y + int(h/2);

		# Now, find where the new region will start (ie, corner point):
		corner_x = center_x - int(scale_factor*w/2);
		corner_y = center_y - int(scale_factor*h/2);

		# Clean the values, in case we are out of bounds:
		# In case too close to 0,0
		corner_x = max(corner_x, 0)
		corner_y = max(corner_y, 0)

		# In case too close to edge of indices:
		outer_corner_x = corner_x+scaled_rect.shape[1];
		outer_corner_y = corner_y+scaled_rect.shape[0];

		if (outer_corner_x >= output_frame.shape[1]):
			outer_corner_x = output_frame.shape[1]-1;
			corner_x = outer_corner_x - scaled_rect.shape[1];

		if (outer_corner_y >= output_frame.shape[0]):
			outer_corner_y = output_frame.shape[0]-1;
			corner_y = outer_corner_y - scaled_rect.shape[0];

		# Now, add in our scaled ROI region
		output_frame[corner_y:(corner_y+scaled_rect.shape[0]), corner_x:(corner_x+scaled_rect.shape[1])] = scaled_rect;

		# Save the scaled ROI region so we can pass it as return value:
		new_rect = (corner_x, corner_y, scaled_rect.shape[1], scaled_rect.shape[0]);
		enlarged_rects.append(new_rect);

	return (output_frame, enlarged_rects);


# magnify_areas_in_frame_grabcut
# Given a frame, and an array of quadruples (x,y,w,h) specifying rectangles, magnifies the
# area specified by the rectangles based on a scaling factor
# Returns a tuple (new_frame, enlarged_rects)
# THIS IS A GRABCUT VERSION OF THIS FUNCTION -- WORKS, BUT VERY SLOW AND NOT VERY GOOD
# DO NOT USE 
def magnify_areas_in_frame_grabcut(frame, array_of_rects):

	# Scaling factor
	scale_factor = 2.0;

	# Copy the frame, because scaling one rectangle up may cover up other regions
	output_frame = np.copy(frame);
	
	# Prepare array of enlarged rects for output:
	enlarged_rects = [];

	# For each region:
	for (x, y, w, h) in array_of_rects:
	
		# To ensure we get the full object, we expand our rectangle to include surrounding
		# image. Any background image that is captured by this should be cut out by
		# GrabCut
		expansion_factor = 0.2
		
		# Modify our rectangle. We need to clean the values to make sure we don't spill
		# over edge of frame
		x = max(0, x-int(w*expansion_factor/2.0));
		y = max(0, y-int(h*expansion_factor/2.0));
		w = min(frame.shape[1]-1, int(w*(1+expansion_factor)));
		h = min(frame.shape[0]-1, int(h*(1+expansion_factor)));
	
		# We want to segment out the ROI (eg, face). For easy of manipulation, we
		# generate the mask on the original image, then upscale the image and mask
	
		# Do Grabcut first to find our mask:
				
		# Mask for selecting out face only (return value of grabcut, needs it passed in as an arg)
		mask = np.zeros(frame.shape[:2], np.uint8);		

		# Some "models" for processing -- just memory allocation
		bgdModel = np.zeros((1,65),np.float64)
		fgdModel = np.zeros((1,65),np.float64)

		# Iteration count for GrabCut
		grabcut_iter_count = 3

		# Now perform GrabCut. Our ROI is within our predetermined rectangle (x,y,w,h)
		cv2.grabCut(frame, mask, (x,y,w,h), bgdModel, fgdModel, grabcut_iter_count, cv2.GC_INIT_WITH_RECT)

		# Mask needs to be processed because it flags "definite" vs "probable" --> simplify
		# Note -- this mask is 2d. We are dealing with frames with 3 color components, so
		# mask will need to be broadcasted to appropriate dimensionality
		mask = np.where((mask==2)|(mask==0),0, 255).astype('uint8');
		
		# Now, select our ROI rectangle and corresponding mask region
		rect = frame[y:y+h, x:x+w];
		rect_mask = mask[y:y+h, x:x+w];

 		#Scale area based on factor
		scaled_rect = cv2.resize(rect, (int(w*scale_factor), int(h*scale_factor)));
		scaled_rect_mask = cv2.resize(rect_mask, (int(w*scale_factor), int(h*scale_factor)));
		
		# Now, need to paste the scaled region on to original
		# We do this by pasting center of scaled region on center of original rectangle 
		
		# First find center point of the rect:
		center_x = x + int(w/2);
		center_y = y + int(h/2);
		
		# Now, find where the new region will start (ie, corner point):
		corner_x = center_x - int(scale_factor*w/2);
		corner_y = center_y - int(scale_factor*h/2);
		
		# Clean the values, in case we are out of bounds:
		# In case too close to 0,0
		corner_x = max(corner_x, 0)
		corner_y = max(corner_y, 0)
		
		# In case too close to edge of indices:
		outer_corner_x = corner_x+scaled_rect.shape[1];
		outer_corner_y = corner_y+scaled_rect.shape[0];
		
		if (outer_corner_x >= output_frame.shape[1]):
			outer_corner_x = output_frame.shape[1]-1;
			corner_x = outer_corner_x - scaled_rect.shape[1];

		if (outer_corner_y >= output_frame.shape[0]):
			outer_corner_y = output_frame.shape[0]-1;
			corner_y = outer_corner_y - scaled_rect.shape[0];
		
		log_flag(debug_flag_info, "Corner X, Y: [{0}, {1}]".format(corner_x, corner_y));
		log_flag(debug_flag_info, "Scaled Rect shape: [{0}, {1}]".format(scaled_rect.shape[1], scaled_rect.shape[0]));
		
		# Now, want to add in our ROI region
		# First, apply mask to the ROI, then the inverse mask to the frame, and then 
		# add it in
		#print("scaled_rect: [{0}, {1}, {2}]".format(scaled_rect.shape[1], scaled_rect.shape[0], scaled_rect.shape[2]));
		#print("scaled_rect_mask: [{0}, {1}, {2}]".format(scaled_rect_mask.shape[1], scaled_rect_mask.shape[0], scaled_rect_mask.shape[2]));

		# Before applying the mask, we need to broadcast it to appropriate dimensionality:
		#broadcasted_mask = np.broadcast_to(scaled_rect_mask, scaled_rect.shape);
		broadcasted_mask = scaled_rect_mask;


		# Python giving issues broadcasting the mask to the appropriate dimensionality, so
		# just apply mask on each dimension (3):
		scaled_rect[:,:,0] = cv2.bitwise_and(scaled_rect[:,:,0], broadcasted_mask);
		scaled_rect[:,:,1] = cv2.bitwise_and(scaled_rect[:,:,1], broadcasted_mask);
		scaled_rect[:,:,2] = cv2.bitwise_and(scaled_rect[:,:,2], broadcasted_mask);
		
		output_frame[corner_y:(corner_y+scaled_rect.shape[0]), corner_x:(corner_x+scaled_rect.shape[1]), 0] = cv2.bitwise_and(output_frame[corner_y:(corner_y+scaled_rect.shape[0]), corner_x:(corner_x+scaled_rect.shape[1]),0], cv2.bitwise_not(broadcasted_mask));
		output_frame[corner_y:(corner_y+scaled_rect.shape[0]), corner_x:(corner_x+scaled_rect.shape[1]), 1] = cv2.bitwise_and(output_frame[corner_y:(corner_y+scaled_rect.shape[0]), corner_x:(corner_x+scaled_rect.shape[1]),1], cv2.bitwise_not(broadcasted_mask));
		output_frame[corner_y:(corner_y+scaled_rect.shape[0]), corner_x:(corner_x+scaled_rect.shape[1]), 2] = cv2.bitwise_and(output_frame[corner_y:(corner_y+scaled_rect.shape[0]), corner_x:(corner_x+scaled_rect.shape[1]),2], cv2.bitwise_not(broadcasted_mask));

		output_frame[corner_y:(corner_y+scaled_rect.shape[0]), corner_x:(corner_x+scaled_rect.shape[1])] += scaled_rect;

		new_rect = (corner_x, corner_y, scaled_rect.shape[1], scaled_rect.shape[0]);

		# And add rectangle to enlarged rect array:
		enlarged_rects.append(new_rect);

		
