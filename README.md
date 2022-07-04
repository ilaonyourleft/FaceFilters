# FaceFilters

Use python2 to run both files, since cv2 and dlib libraries have stable functions only for python 2.7.

final.py capture the video from the webcam and it applies the filter selected by command line. The .gif file is then saved in the 'new' directory, with name 'output_panda.gif' or 'output_pika.gif' or 'output_eyes.gif'. The file is overwritten if the script is launched a second time. 

imageprocessing-filters.py applies a series of filters directly on the video stream of the webcam.

Both scripts get the filters from the 'png' directory.

haarcascade_frontalface_default.xml: in opencv, it is a classifier trained to identify object of a specified kind - in this case, frontalface.

shape_predictor_68_face_landmarks.dat: it is a trained network to recognize the 68 landmarks of the analyzed face.
