###############################################################################
# output_frame_stream.py
#
# Description:
#	Class definition for output frame stream object. Handles:
#		- Interface with output stream (abstracts it away from user)
#		- Output frame stream, including writing to file, etc
#
#	Currently set up to only write output frames to a file in MJPEG format
#
#	To Do:
#		- Allow for camera (eg, webcam) streams for use in piping into Unity, etc
#
###############################################################################

# Import necessary packages
import cv2
import sys
import numpy as np

class Output_Frame_Stream:

	# Initializer method:
	def __init__(self, output_video_name, frame_size_x, frame_size_y):
	
		# Set up output video stream:
		fourcc_code = cv2.VideoWriter_fourcc(*'MJPG');
		self.output_video = cv2.VideoWriter(output_video_name, fourcc_code, 25.0, (frame_size_x, frame_size_y));
		
	# Method to write frame to output stream:
	def write_output_frame(self, frame):
		self.output_video.write(frame);
		
	# Close stream when we're done:
	def close_stream(self):
		self.output_video.release();