###############################################################################
# main_loop.py
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

# Import some helper packages:
import argparse
import numpy as np
from collections import deque

# Import our own written modules:
from includes.frame_stream import Frame_Stream
from includes.face_tracker import Face_Tracker
from includes.retargeter import Retargeter
from includes.logger import Logger
from includes.image_funcs import Image_Funcs

# For testing - flag to enable test frames to show
show_debug_frames = True;

# For testing - number of frames to retarget before testing 
num_frames_to_retarget = 500;

# Number of seams to be removed:
num_seams_to_remove_vertical = 100;
num_seams_to_remove_horizontal = 0;

###############################################################################
# Main Loop

# Overall steps:
# 1. Get input video, load it appropriately (initializes all vars)
# 2. Iterate through each frame:
# -- a. Preprocess frame (included ROI/face detection, magnification)
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

# Instantiate some the objects we'll need:

# Load input video into a new frame_stream object:
input_video = Frame_Stream(args["video_path"]);

# Set up face tracker object to keep track of faces:
roi_detect_rate = 10; # detect ROIs every 10 frames
face_tracker_list = Face_Tracker(args["cascade"], roi_detect_rate);

# Set up the logger:
msg_logger = Logger.get_logger();
msg_logger.set_logger_level(Logger.DEBUG_FLAG_TIME);


# Frame queue for displaying:
original_video_queue = deque([]);
retarget_video_queue = deque([]);


# Grab first frame to initialize. Fail if opening/reading causes error:
if (input_video.more_frames_available()):
	input_video.get_next_frame();
else:
	print("Video opening failed on initialization");
	sys.exit(1);

# Set up our retargeter for frame manipulation:
# First, let's get the frame size:
frame_size_x = input_video.get_current_frame().shape[1];
frame_size_y = input_video.get_current_frame().shape[0];

# Now, set up retargeter:
frame_retargeter = Retargeter(frame_size_x, frame_size_y, num_seams_to_remove_vertical, num_seams_to_remove_horizontal);


# Setup stream for output video to be saved
# Create filename for output video:
# First, get only file name:
video_file_name = args["video_path"].split("/")[-1];
output_video_name = "retargeted_" + video_file_name.split(".")[0] + ".avi";

# Grab dimensions of output video:
output_x = frame_size_x - num_seams_to_remove_vertical;
output_y = frame_size_y - num_seams_to_remove_horizontal;

# Open the output video stream
fourcc_code = cv2.VideoWriter_fourcc(*'MJPG');
output_video = cv2.VideoWriter(output_video_name, fourcc_code, 25.0, (output_x, output_y));




while (input_video.more_frames_available() and (len(retarget_video_queue) < num_frames_to_retarget)):

	# Measure processing time per frame
	msg_logger.log_msg(Logger.DEBUG_FLAG_TIME, "Starting frame {0}".format(len(retarget_video_queue)+1));
	msg_logger.log_time('Total frame', Logger.TIME_START);

	# First, update our frames
	input_video.get_next_frame();
	

	# Preprocess frame
	# This includes ROI detection and any object magnification
	# Returns a modified frame and mask corresponding to ROIs
	
	# Detect/track faces:
	msg_logger.log_time('Face detection/tracking', Logger.TIME_START);
	face_tracker_list.refresh_face_trackers(input_video.get_current_frame_gray());
	msg_logger.log_time('Face detection/tracking', Logger.TIME_END);

	# Now, based on faces, magnify those areas. We then modify the frame stream to keep
	# that new frame:
	
	# Grab the current frame, and the detected faces:
	current_frame = input_video.get_current_frame();
	roi_rects = face_tracker_list.get_tracked_faces();
	
	# Then, magnify those faces. We get a new set of rectangles corresponding to the
	# magnified regions
	msg_logger.log_time('Face magnification', Logger.TIME_START);	
	(current_frame, roi_rects) = Image_Funcs.magnify_areas_in_frame(current_frame, roi_rects);
	msg_logger.log_time('Face magnification', Logger.TIME_END);

	# Now that we've modified our frame, give it back to frame_stream:
	input_video.set_current_frame(current_frame);

	# Calculate the frame's energy map:
	msg_logger.log_time('Energy map', Logger.TIME_START);
	energy_map = frame_retargeter.generate_energy_map(input_video.get_previous_frame_gray(), input_video.get_current_frame_gray(), roi_rects);
	msg_logger.log_time('Energy map', Logger.TIME_END);
	
	# Then, seam carve our current frame:
	msg_logger.log_time('Carving frame', Logger.TIME_START);
	output_frame = frame_retargeter.retarget_frame(input_video.get_current_frame());
	msg_logger.log_time('Carving frame', Logger.TIME_END);

	# Convert frame to a normal byte range in preparation for saving
	output_frame = (output_frame * 255.0).astype('u1')
	
	# Finally, save the retargeted frame:
	output_video.write(output_frame);
	
	# and save frames in queue to play after processing:
	original_video_queue.append(input_video.get_current_frame());
	retarget_video_queue.append(output_frame);
	
	# Log times:
	msg_logger.log_time('Total frame', Logger.TIME_END);
	msg_logger.log_msg(Logger.DEBUG_FLAG_TIME,"===============================================");
	
	if (show_debug_frames):
		cv2.imshow("Energy Map", energy_map);
		cv2.imshow("Carved Frame", output_frame);
		cv2.waitKey(40);
	
# Now that we're all done, release our videos:
input_video.close_stream();
output_video.release();
cv2.destroyAllWindows();

# Now play the frames:
while (retarget_video_queue):
	cv2.imshow('Original video', original_video_queue.popleft());
	cv2.imshow('Retargeted video', retarget_video_queue.popleft());
	cv2.waitKey(40); # Assumes 25 frames/sec video;