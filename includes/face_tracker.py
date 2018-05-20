###############################################################################
# face_tracker.py
#
# Description:
#	Class definition for face trackers. Implements face detect algorithms based
#	on argument frames, and then keeps track of them using a face tracker 
#	(using dlib correlation tracker). Thus, this module can trace multiple
#	faces.
#
#	Main usage: calling "refresh_face_trackers" and "get_tracked_faces"
#
#	To Do:
#		
###############################################################################

# Import necessary packages
import cv2
import sys
import dlib

# Import some helper packages:
import argparse
import numpy as np

class Face_Tracker:

	# Class variables (static)
	# Need to save face detection cascade module statically:
	face_detection_cascade = None;

	# Initializer method:
	def __init__(self, cascade_path, detect_rate = 10):
		
		# Initialize cascade as static variable:
		Face_Tracker.face_detection_cascade = cv2.CascadeClassifier(cascade_path)
		
		# Initialize some object variables:
		self.list_of_trackers = [];
		
		# Specify detection vs tracking rate:
		self.detection_rate = detect_rate
		self.frame_num_in_cycle = 0; # frame count for detection iterations


	# detect_faces_in_frame
	# Helper function -- given a frame, detects faces using the static variable cascade
	# Returns tuple of (x,y,w,h) specifying faces
	def detect_faces_in_frame(frame):

		# Detect faces in the image
		faces = Face_Tracker.face_detection_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20));

		return faces;


	# face_trackers_to_rectangle_tuples
	# This is the main return function for the object -- returns the faces tracked as
	# tuples (x,y,w,h)
	def get_tracked_faces(self):
		return Face_Tracker.face_trackers_to_rectangle_tuples(self.list_of_trackers);


	# Main "refresh" function for the face trackers
	# Either continues to track the known faces, or detects new ones (based on the rate)
	# TAKES IN GRAYSCALE FRAME
	def refresh_face_trackers(self, frame):

		# Detect faces if its time for it, otherwise continue tracking known faces:
		if ((self.frame_num_in_cycle % self.detection_rate) == 0):
			self.detect_new_faces(frame);
		else:
			self.continue_tracking_faces(frame);

		# Advance the frame counter:
		self.frame_num_in_cycle = (self.frame_num_in_cycle+1) % self.detection_rate


	# Refreshes the object's list of trackers. Deletes those with low quality.
	def continue_tracking_faces(self, frame):

		threshold_quality = 7;

		#Update all the trackers and remove the ones for which the update
		#indicated the quality was not good enough
		trackers_to_delete = []  
	
		for tracker in self.list_of_trackers:  
			tracking_quality = tracker.update( frame )

			#If the tracking quality is good enough, we must delete
			#this tracker
			if tracking_quality < threshold_quality:
				trackers_to_delete.append( tracker )

		for tracker in trackers_to_delete:
			self.list_of_trackers.remove( tracker )



	# Method to newly detect faces. Any new faces are added as new trackers.
	def detect_new_faces(self, frame):

		# First, detect all faces in frame:
		faces = Face_Tracker.detect_faces_in_frame(frame);

		#Loop over all faces 
		for (_x,_y,_w,_h) in faces:  

			#We need to convert it to int here because of the
			#requirement of the dlib tracker. If we omit the cast to
			#int here, you will get cast errors since the detector
			#returns numpy.int32 and the tracker requires an int
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
			for tracker in self.list_of_trackers:
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

			#If no matched fid, then we have found a new face. Make a new tracker
			if matchedFid is None:

				#Create and store the tracker 
				new_tracker = dlib.correlation_tracker()
				new_tracker.start_track(frame,
									dlib.rectangle( x,
													y,
													x+w,
													y+h))

				self.list_of_trackers.append(new_tracker);

	# face_trackers_to_rectangle_tuples
	# Given array of face trackers, returns faces as a tuple of (x,y,w,h)
	@staticmethod
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
