# low_vision_retargeting

==== DESCRIPTION: =======================================================================

This script is a video processing script meant for prototyping visual enhancements for
low vision patients

==== REQUIREMENTS: ======================================================================

Python 3.5.5 or higher (Anaconda)
OpenCV 3.1.0

==== HOW TO USE: ========================================================================

Call script with required arguments at command line:

$ python -v <input_video_path> -c <face_detection_cascade_file_path>

The script will run and will output the retargeted video
"retargeted_<input_video_file_name>.avi" in MJPEG format within the same directory.

There a few modifiable flags within the script to allow for easy usage. They are all 
found at the top of the script file. They are all mostly self-explanatory.

==== TO DOS: ===========================================================================

- Fix grabcut of face
- Optimize retargeting algorithm to minimize jitter
- Convert script to OOP (maintaining state between frames is hard with functional programming)
- Differential face magnification based on who is talking

