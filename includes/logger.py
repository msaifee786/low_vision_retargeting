###############################################################################
# logger.py
#
# Description:
#	Class defintion for logger object. Logs error/info/timing messages. To use
#	the logger, just call on the class method get_logger and use that object.
#	Do not instantiate your own.
#
#
#	To Do:
#
###############################################################################

# Import necessary packages
import sys

# Import some helper packages:
import math
import time

class Logger:

	# Need a class variable to hold the logger instance
	# We want logger to be available globally, so we hold onto logger as a class var and
	# access it using a class method
	global_logger = None;
	
	# Debug flag variables
	DEBUG_FLAG_NONE = -1;
	DEBUG_FLAG_TIME = 0;
	DEBUG_FLAG_INFO = 1;
	DEBUG_FLAG_WARNING = 2;

	# Debug flag message prefixes:
	debug_flag_levels = {
	  DEBUG_FLAG_TIME: '[TIME]',
	  DEBUG_FLAG_INFO: '[INFO]',
	  DEBUG_FLAG_WARNING: '[WARNING]'
	  }; 

	# Default flag:
	DEFAULT_FLAG_LEVEL = DEBUG_FLAG_TIME;

	# Flags for use in time benchmarking
	TIME_START = 0;
	TIME_END = 1;

	# Initializer method:
	def __init__(self, flag_level):
		
		# Set up a dictionary to hold onto any benchmarking times:
		self.myTimes = {};
		
		# Set up flag level:
		self.myFlagLevel = flag_level;
		
	# Message to log any message:
	def log_msg(self, flag_level, msg):

		# Do we need report this?
		if (flag_level <= self.myFlagLevel):
	
			# Grab prefix string:
			prefix_string = Logger.debug_flag_levels[flag_level];
		
			# Construct our display string:
			display_string = prefix_string + " " + msg;
		
			# And print it:
			print(display_string);
	
	def log_time(self, event_string, time_flag):
	
		# If we're starting to time an event, just log the time and finish:
		if (time_flag == Logger.TIME_START):
			self.myTimes[event_string] = time.time();

		# If we're ending a time of an event, pull the old time and log the difference
		# If no start time was logged, it defaults to now:
		if (time_flag == Logger.TIME_END):
			# Grab start time. Default value current time (if key is missing)
			start_time = self.myTimes.get(event_string, time.time())
			
			# Get end time (now):
			end_time = time.time();
	
			# Time elapsed in ms:
			time_elapsed = math.trunc(1000*(end_time - start_time));
	
			# Log the time:
			self.log_msg(Logger.DEBUG_FLAG_TIME, event_string + " done in {0} ms".format(time_elapsed));
	
	@classmethod
	def get_logger(cls):
		# Class method to access the logger object. All modules using the logger should
		# only get loggers through this method -- this will allow all modules to use the
		# same logger.
		
		# If we don't have one yet, make one:
		if (Logger.global_logger is None):
			Logger.global_logger = Logger(Logger.DEFAULT_FLAG_LEVEL)
	
		return Logger.global_logger;
		
	@classmethod
	def set_logger_level(cls, level):
		Logger.get_logger().myFlagLevel = level
		