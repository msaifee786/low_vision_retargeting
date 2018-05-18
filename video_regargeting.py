###############################################################################
# video_seam_carving.py
#
# Description:
#	Given a video, seam carves it frame by frame and outputs a retargeted video
#
# Inputs:
#	Input video
#
# Outputs:
#	Retargeted (seam carved) version of input video
#
# To Do:
#	- Optimize face detection
#	- GrabCut face
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

# Debug flag variables
debug_flag_none = -1;
debug_flag_time = 0;
debug_flag_info = 1;
debug_flag_warning = 2;

debug_flag_levels = {
  debug_flag_time: '[TIME]',
  debug_flag_info: '[INFO]',
  debug_flag_warning: '[WARNING]'
  }; 

debug_flag = debug_flag_time;

# For testing - flag to enable test frames to show
show_debug_frames = True;

# For testing - number of frames to retarget before testing 
num_frames_to_retarget = 1000;

# Number of seams to be removed:
num_seams_to_remove_vertical = 0;
num_seams_to_remove_horizontal = 0;

# How often (number of frames) to refresh face tracker
refresh_face_rate = 10





###############################################################################
# Helper functions:

# log_flag
# Given a flag level, will print string + value if the flag priority is lower than the 
# global debug flag level.
def log_flag (flag_level, string):

	# Import global debug flag:
	global debug_flag;
	global debug_flag_levels;

	# Do we need report this?
	if (flag_level <= debug_flag):
	
		# Grab prefix string:
		prefix_string = debug_flag_levels[flag_level];
		
		# Construct our display string:
		display_string = prefix_string + " " + string;
		
		# And print it:
		print(display_string);


# update_energy_map
# Given previous and current frames (and current energy map), finds "energy" in frame
# for use in seam carving
# Notably, assumes frames are Numpy arrays. See below for implementation
def update_energy_map(previous_frame, current_frame, faces):

	# TODO:
	# Fix weighting between temporal and spatial energy
	
	
	# Overall steps:
	# 1. Calculate spatial energy (via simple filtering)
	# 2. Facial recognition --> increase energy of faces (to preserve them on carving)
	# 3. Calculate temporal frequency (via helper function)
	# 4. Add spatial + temporal frequency for final value
	
	
	# Get spatial energy via edge detection filter
	start_ts_filter = time.time();
	mag = calculate_spatial_energy(current_frame);
	#mag = cv2.Canny(current_frame, 100, 200);
	mag = cv2.dilate(mag, None, iterations=1);

	time_elapsed_filter = time.time() - start_ts_filter;

	if (show_debug_frames):
		cv2.imshow("Spatial energy", mag);

	# Get average spatial energy for later comparison/logging:
	average_spatial_energy = np.mean(mag);

	# Now, increase energy of faces:
	start_ts_faces = time.time();
	for (x, y, w, h) in faces:
		mag[y:(y+h), x:(x+w)] = 255;
	
	time_elapsed_faces = time.time() - start_ts_faces;


	# Now, deal with temporal frequency energy:
	start_ts_temp_freq = time.time();
	temp_freq_energy = calculate_temporal_energy(current_frame, previous_frame);
	time_elapsed_temp_freq = time.time() - start_ts_temp_freq;


	# Get average temporal energy:
	average_temp_energy = np.mean(temp_freq_energy);

	# Add temporal frequency to spatial to get our full energy:
	mag = mag + temp_freq_energy;

	# Report any flags:
	global debug_flag_time, debug_flag_info;

	log_flag(debug_flag_time, "Spatial energy filter in {0} ms".format(math.trunc(1000*time_elapsed_filter)));
#	log_flag(debug_flag_time, "Face energy in {0} ms".format(math.trunc(1000*time_elapsed_faces)));
	log_flag(debug_flag_time, "Temporal energy in {0} ms".format(math.trunc(1000*time_elapsed_temp_freq)));
	
	log_flag(debug_flag_info, "Average Spatial Freq Value: {0}".format(average_spatial_energy));
	log_flag(debug_flag_info, "Average Temp Freq Value: {0}".format(average_temp_energy));

	if (show_debug_frames):
		cv2.imshow("Temporal energy", temp_freq_energy);
		cv2.imshow("Total energy map", mag);

	return mag;






###############################################################################
# Energy calculation functions:

# calculate_spatial_energy
# Given a (grayscale) frame, calculates median value of pixels, then performs canny filter
# on image with min/max thresholds based on a "sigma value" distance away from median.
# From www.pyimagesearch.com
def calculate_spatial_energy(frame):

	# Canny filter sigma coefficient (determines how narrow/wide to threshold)
	canny_sigma=0.33

	# compute the median of the single channel pixel intensities
	v = np.median(frame)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - canny_sigma) * v))
	upper = int(min(255, (1.0 + canny_sigma) * v))
	edged = cv2.Canny(frame, 100, 200)
 
	# return the edged image
	return edged


# calculate_temporal_energy
# Given current and previous frame, calculates temporal energy
# First we blur, then take absolute difference between frames and threshold based on a 
# cutoff value. Then we dilate those areas to make continuous contours
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


# get_first_energy_map
# Returns an array of 0, based on size of rame
def get_first_energy_map(frame):

	return np.zeros(frame.shape)








###############################################################################
# ROI detection functions

# detect_faces_in_frame
# Given a frame, detects faces using global variable cascade. Returns array of x,y,w,h
# specifying faces
def detect_faces_in_frame(frame):

	global faceCascade;

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20));

	return faces;

# renew_face_trackers
# Given an array of face trackers, detects faces in frame and adds new trackers based on it
def look_for_new_faces(face_trackers, frame):

	# First, detect all faces in frame:
	faces = detect_faces_in_frame(frame);

	#Loop over all faces and check if the area for this
	#face is the largest so far
	#We need to convert it to int here because of the
	#requirement of the dlib tracker. If we omit the cast to
	#int here, you will get cast errors since the detector
	#returns numpy.int32 and the tracker requires an int
	for (_x,_y,_w,_h) in faces:  
		x = int(_x)
		y = int(_y)
		w = int(_w)
		h = int(_h)

		#calculate the centerpoint
		x_bar = x + 0.5 * w
		y_bar = y + 0.5 * h

		#Variable holding information which faceid we 
		#matched with
		matchedFid = None

		#Now loop over all the trackers and check if the 
		#centerpoint of the face is within the box of a 
		#tracker
		for tracker in face_trackers:
			tracked_position =  tracker.get_position()

			t_x = int(tracked_position.left())
			t_y = int(tracked_position.top())
			t_w = int(tracked_position.width())
			t_h = int(tracked_position.height())

			#calculate the centerpoint
			t_x_bar = t_x + 0.5 * t_w
			t_y_bar = t_y + 0.5 * t_h

			#check if the centerpoint of the face is within the 
			#rectangleof a tracker region. Also, the centerpoint
			#of the tracker region must be within the region 
			#detected as a face. If both of these conditions hold
			#we have a match
			if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
				 ( t_y <= y_bar   <= (t_y + t_h)) and 
				 ( x   <= t_x_bar <= (x   + w  )) and 
				 ( y   <= t_y_bar <= (y   + h  ))):
				matchedFid = True

		#If no matched fid, then we have to create a new tracker
		if matchedFid is None:

			#Create and store the tracker 
			new_tracker = dlib.correlation_tracker()
			new_tracker.start_track(frame,
								dlib.rectangle( x,
												y,
												x+w,
												y+h))

			face_trackers.append(new_tracker);


# refresh_face_trackers
# Given an array of face trackers, refreshes them. Deletes any with low quality.
def refresh_face_trackers(face_trackers, frame):

	threshold_quality = 7;

	#Update all the trackers and remove the ones for which the update
	#indicated the quality was not good enough
	trackers_to_delete = []  
	
	for tracker in face_trackers:  
		tracking_quality = tracker.update( frame )

		#If the tracking quality is good enough, we must delete
		#this tracker
		if tracking_quality < threshold_quality:
			trackers_to_delete.append( tracker )

	for tracker in trackers_to_delete:
		face_trackers.remove( tracker )


# face_trackers_to_rectangle_tuples
# Given array of face trackers, returns faces as a tuple of (x,y,w,h)
def face_trackers_to_rectangle_tuples(face_trackers):

	faces = [ ];

	for tracker in face_trackers:
		tracked_position =  tracker.get_position()

		t_x = int(tracked_position.left())
		t_y = int(tracked_position.top())
		t_w = int(tracked_position.width())
		t_h = int(tracked_position.height())

		face = (t_x, t_y, t_w, t_h);
		
		faces.append(face);
	
	return faces;




###############################################################################
# Frame preprocessing functions


# preprocess_frame
# Given a frame, will detect ROIs (faces, etc), then will differentially magnify those
# objects in the frame. Returns modified frame and a list of the rectangles of ROIs (for
# use in energy calculations
def preprocess_frame(frame, frame_count):

	# Need to pull in (static) global face_trackers, because we need to keep track of it
	# from frame to frame
	global face_trackers;

	# TODO:
	# - Grab cut faces/ROI
	# - Magnify faces
	# - Mask function for ROIs, AND together

	# Need to keep track of our ROIs/masks
	
	# First, need to find ROIs
	# Faces:
	
	start_ts_face_tracking = time.time();
	
	# We track faces. We do a full detection every so often to pick up any new faces in 
	# the frame
	# Check if we're due for a full detection, 
	
	# We need grayscale version of photo:
	frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
	
	if (len(retarget_video_queue) % refresh_face_rate == 1):
		look_for_new_faces(face_trackers, frame_gray);
	else:
	# Otherwise, refresh trackers
		refresh_face_trackers(face_trackers, frame_gray);

	elapsed_time_face_tracking = time.time() - start_ts_face_tracking;
	log_flag(debug_flag_time, "Face tracking in {0} ms".format(math.trunc(1000*elapsed_time_face_tracking)));

		
	# Pull the regions as rectangles:
	faces = face_trackers_to_rectangle_tuples(face_trackers);

	# Magnify faces:
	start_ts_mag_faces = time.time();
	
	# Then, modify current frame to magnify them
	frame, faces = magnify_areas_in_frame(frame, faces);
	
	elapsed_time_mag_faces = time.time() - start_ts_mag_faces;
	log_flag(debug_flag_time, "Magnified faces in {0} ms".format(math.trunc(1000*elapsed_time_mag_faces)));


	return frame, faces


# magnify_areas_in_frame
# Given a frame, and an array of quadruples (x,y,w,h) specifying rectangles, magnifies the
# area specified by the rectangles based on a scaling factor
# Returns a tuple (new_frame, enlarged_rects)
# NO ERROR CHECKING DONE TO MAKE SURE THAT SCALED REGION WILL FIT IN FRAME
def magnify_areas_in_frame(frame, array_of_rects):

	# Scaling factor
	scale_factor = 2.0;

	# Copy the frame, because scaling one rectangle up may cover up other regions
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

		log_flag(debug_flag_info, "Corner X, Y: [{0}, {1}]".format(corner_x, corner_y));
		log_flag(debug_flag_info, "Scaled Rect shape: [{0}, {1}]".format(scaled_rect.shape[1], scaled_rect.shape[0]));

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


###############################################################################
# Main Loop

# Overall steps:
# 1. Get input video, load it appropriately (initializes all vars)
# 2. Iterate through each frame:
# -- a. Preprocessor frame (included ROI detection, magnification)
# -- b. Calculate energy map (based on current and previous frame)
# -- c. Seam carve frame
# -- d. Save current frame to video


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video_path", required=True,
	help="path to input image file")
ap.add_argument("-c", "--cascade", required=True,
	help="path to cascade xml file for object ROI detection")
ap.add_argument("-d", "--debug", type=str,
	default="false", help="seam removal direction")
args = vars(ap.parse_args())

# Load input video:
video = cv2.VideoCapture(args["video_path"]);

# Initialize all variables
# Frames:
previous_frame_gray = 0;
current_frame = 0;
current_frame_gray = 0;

# Frame count (for keeping track of when to detect vs track ROI objects)
frame_count = 0;
roi_detect_rate = 10; # detect ROIs every 10 frames

# Frame queue for displaying:
original_video_queue = deque([]);
retarget_video_queue = deque([]);

# Create the cascade
# NOTE: this is a global variable, to be used in face detections
faceCascade = cv2.CascadeClassifier(args["cascade"])

# Face tracker array:
# NOTE: this is a global variable, to be used in face detections/tracking
face_trackers = [];



# Grab first frame to initialize. Fail if opening/reading causes error:
if (video.isOpened()):
	ret, current_frame = video.read()
	current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
	previous_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
else:
	print("Video opening failed on initialization");
	sys.exit(1);

if (not ret):
	print("Error on reading first frame");
	sys.exit(1);


# Setup stream for output video to be saved
# Create filename for output video:
# First, get only file name:
video_file_name = args["video_path"].split("/")[-1];

output_video_name = "retargeted_" + video_file_name.split(".")[0] + ".avi";

# Grab dimensions of output video:
output_x = current_frame.shape[1] - num_seams_to_remove_vertical;
output_y = current_frame.shape[0] - num_seams_to_remove_horizontal;

# Open the output video stream
fourcc_code = cv2.VideoWriter_fourcc(*'MJPG');
output_video = cv2.VideoWriter(output_video_name, fourcc_code, 25.0, (output_x, output_y));

# Get first energy map:
energy_map = get_first_energy_map(current_frame_gray);

while (video.isOpened() and (len(retarget_video_queue) < num_frames_to_retarget)):

	# Measure processing time per frame
	log_flag(debug_flag_time, "Starting frame {0}".format(len(retarget_video_queue)+1))
	start_ts_frame = time.time();

	# First, update our frames
	# Update previous frame, pull new frame down, calculate gray version
	# Also increment our frame count (mod detection rate)
	previous_frame_gray = current_frame_gray;
	ret, current_frame = video.read();
	current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
	frame_count = (frame_count+1) % roi_detect_rate

	# Preprocess frame
	# This includes ROI detection and any object magnification
	# Returns a modified frame and mask corresponding to ROIs
	current_frame, roi_rects = preprocess_frame(current_frame, frame_count);
	current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)


# 	Update our face trackers:
# 	start_ts_face_tracking = time.time();
# 	if (len(retarget_video_queue) % refresh_face_rate == 0):
# 		look_for_new_faces(face_trackers, current_frame_gray);
# 	else:
# 		refresh_face_trackers(face_trackers, current_frame_gray);
# 		
# 	faces = face_trackers_to_rectangle_tuples(face_trackers);
# 
# 	elapsed_time_face_tracking = time.time() - start_ts_face_tracking;
# 	log_flag(debug_flag_time, "Face tracking in {0} ms".format(math.trunc(1000*elapsed_time_face_tracking)));
# 
# 	Magnify faces:
# 	start_ts_mag_faces = time.time();
# 	
# 	Then, modify current frame to magnify them (update gray version too)
# 	current_frame, faces = magnify_areas_in_frame(current_frame, faces);
# 	current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
# 	
# 	elapsed_time_mag_faces = time.time() - start_ts_mag_faces;
# 	log_flag(debug_flag_time, "Magnified faces in {0} ms".format(math.trunc(1000*elapsed_time_mag_faces)));


	#start_ts_face_recog = time.time();	
	#faces = detect_faces_in_frame(current_frame_gray);
	#elapsed_time_face_recog = time.time() - start_ts_face_recog;
	#log_flag(debug_flag_time, "Facial recognition in {0} ms".format(math.trunc(1000*elapsed_time_face_recog)));
	



	# Update our energy map:
	energy_map = update_energy_map(previous_frame_gray, current_frame_gray, roi_rects);
	
	
	
	# Then, seam carve our current frame:
	start_ts_carve = time.time();
	carved = transform.seam_carve(current_frame, energy_map, "vertical", num_seams_to_remove_vertical)
	#carved = transform.seam_carve(carved, energy_map, "horizontal", num_seams_to_remove_horizontal)
	elapsed_time_carve = time.time() - start_ts_carve;
	log_flag(debug_flag_time, "Seam carving in {0} ms".format(math.trunc(1000*elapsed_time_carve)));

	# Convert frame to a normal byte range in preparation for saving
	carved = (carved * 255.0).astype('u1')
	
	# Finally, save our current frame to our queue to be committed later:
	output_video.write(carved);
	original_video_queue.append(current_frame);
	retarget_video_queue.append(carved);
	
	time_elapsed_frame = time.time() - start_ts_frame;
	
	# Log times:
	log_flag(debug_flag_time, "Retargeted frame {0} in {1} ms".format(len(retarget_video_queue), math.trunc(1000*time_elapsed_frame)));
	log_flag(debug_flag_time, "===============================================");
	
	
	if (show_debug_frames):
		cv2.imshow("Carved Frame", carved);
		cv2.waitKey(40);
	
# Now that we're all done, release our video:
video.release();
output_video.release();
cv2.destroyAllWindows();

# Now commit our saved retargeted video: (or, play it)
while (retarget_video_queue):
	cv2.imshow('Original video', original_video_queue.popleft());
	cv2.imshow('Retargeted video', retarget_video_queue.popleft());
	cv2.waitKey(40); # Assumes 25 frames/sec video