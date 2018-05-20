###############################################################################
# frame_stream.py
#
# Description:
#	Class definition for frame stream object. Handles:
#		- Interface with input stream (abstracts it away from user)
#		- Incoming frame stream, including current/previous frame
#
#	Currently set up to only handle input videos (not cameras, etc)
#
#	To Do:
#		- Allow for camera (eg, webcam) streams
#
###############################################################################

# Import necessary packages
import cv2
import sys
import numpy as np

class Frame_Stream:

	# Initializer method:
	def __init__(self, video_path):
	
		# Load input video:
		self.video = cv2.VideoCapture(video_path);
		
		# Initialize variables:
		self.current_frame = None;
		self.current_frame_gray = None;
		self.previous_frame = None;
		self.previous_frame_gray = None;
		
	# Boolean wrapper to see if more frames available from stream:
	def more_frames_available(self):
		return self.video.isOpened();
	
	# Method to "advance" along stream, meaning pull a new frame from the stream and 
	# advance previous frame
	# ASSUMES THAT more_frames_available RETURNS TRUE (NO ERROR CHECKING DONE)
	def get_next_frame(self):

		# First, update previous frame data so we don't lose it:
		self.previous_frame = self.current_frame;
		self.previous_frame_gray = self.current_frame_gray;
	
		# Pull next frame down:
		ret, self.current_frame = self.video.read()
		self.current_frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY);
		
		return self.current_frame;

	# Some simple accessor/mutator functions:
	def get_current_frame(self):
		return self.current_frame
	
	def get_current_frame_gray(self):		
		return self.current_frame_gray;
		
	def get_previous_frame(self):
		return self.previous_frame;
	
	def get_previous_frame_gray(self):
		return self.previous_frame_gray;

	# We edit frames when we magnify objects, etc
	def set_current_frame(self, frame):
		self.current_frame = frame;
		self.current_frame_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY);
		
	# For when we're done with the stream
	def close_stream(self):
		self.video.release()