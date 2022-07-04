import time
import numpy as np
import cv2
import sys
import dlib
from PIL import Image
import os
import glob
import math
from scipy.spatial import distance as dist

print
filterChoice = raw_input("choose the filter you want for your video: insert p for the panda, k for pikachu, e for the eyes: ")
print
print "type q when you want the camera to stop recording. the stream will start in 5 seconds."
if filterChoice == "p" or filterChoice == "k":
	print "and remember to raise your eyebrows!!"
elif filterChoice == "e":
	print "and remember to open your mouth!!"
print
time.sleep(5)

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

#faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

filterPanda_down = 'png/panda_down.png'
filterPanda_up = 'png/panda_up.png'
filterPikachu_down = 'png/pika_down.png'
filterPikachu_up = 'png/pika_up.png'
filterEyes_closed = 'png/mouthClosed.png'
filterEyes_open = 'png/mouthOpen.png'

gif_name = "output"

def create_video():
	# Create a VideoCapture object
	cap = cv2.VideoCapture(0)
	 
	# Check if camera opened successfully
	if (cap.isOpened() == False): 
		print("Unable to read camera feed")
	 
	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
	# We convert the resolutions from float to integer.
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	 
	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
	 
	while(True):
		ret, frame = cap.read()
	 
		if ret == True: 
	     
			# Write the frame into the file 'output.avi'
			out.write(frame)

			# Display the resulting frame    
			cv2.imshow('frame',frame)

			# Press Q on keyboard to stop recording
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	 
		# Break the loop
		else:
			break 
	 
	# When everything done, release the video capture and video write objects
	cap.release()
	out.release()
	 
	# Closes all the frames
	cv2.destroyAllWindows()
	
create_video()
vidcap = cv2.VideoCapture("outpy.avi")
success, frame = vidcap.read()
count = 0
height = []
width = []
ang = []

def open_brows_filter(face_width, dis_right, dis_left, n):
	threshold = (31.4 * face_width) / 152
	#print "threshold " + str(threshold)
	#print
	
	#if (dis_right > threshold and dis_left > threshold):
	if (dis_left > threshold):
		if n == "p":
			fil = Image.open(filterPanda_up)
		elif n == "k":
			fil = Image.open(filterPikachu_up)
	else:
		if n == "p":
			fil = Image.open(filterPanda_down)
		elif n == "k":
			fil = Image.open(filterPikachu_down)
			
	return fil
	
def open_mouth_filter(face_width, dis, n):
	threshold = (8 * face_width) / 152
	#print "threshold " + str(threshold)
	#print
	
	if dis > threshold:
		fil = Image.open(filterEyes_open)
	else:
		fil = Image.open(filterEyes_closed)
		
	return fil

def paste_pattern_brows(n, i, original, ang, width, height, nose_x, nose_y):
	pattern = Image.open("resized_%d.png" % i)
	
	w, h = pattern.size
	
	if n == "p":
		xr = math.cos(ang) * 250 - math.sin(ang) * 325
		yr = math.sin(ang) * 250 + math.cos(ang) * 325
		
		n_x = (xr * w) / width
		n_y = (yr * h) / height
		
		x = nose_x - 1.1 * n_x
		y = nose_y - 0.9 * n_y
	elif n == "k":
		xr = math.cos(ang) * 214 - math.sin(ang) * 468
    		yr = math.sin(ang) * 214 + math.cos(ang) * 468
    		
    		n_x = (xr * w) / width
		n_y = (yr * h) / height
		
		x = nose_x - n_x - 0.26 * n_x
		y = nose_y - n_y

	area = (int(x), int(y))
	original.paste(pattern, area, pattern)
	original.save(str(f))
	
def paste_pattern_mouth(n, i, original, ang, width, height, inner_eye_right_x, inner_eye_right_y, inner_eye_left_x, inner_eye_left_y):
	pattern = Image.open("resized_%d.png" % i)
	
	w, h = pattern.size
	
	dis_eye = dist.euclidean((inner_eye_right_x, inner_eye_right_y), (inner_eye_left_x, inner_eye_left_y))
	
	if n == "e":
		xr = math.cos(ang) * 184 - math.sin(ang) * 99
		yr = math.sin(ang) * 184 + math.cos(ang) * 99
		
		n_x = (xr * w) / width
		n_y = (yr * h) / height
		
		x = inner_eye_right_x - 0.75 * n_x
		y = inner_eye_right_y - n_y
	
	area = (int(x), int(y))
	original.paste(pattern, area, pattern)
	original.save(str(f))

def estimate_distance_brows(sixteen_x, sixteen_y, zero_x, zero_y, eye_right_x, eye_right_y, brow_right_x, brow_right_y, eye_left_x, eye_left_y, brow_left_x, brow_left_y):
	face_width = dist.euclidean((sixteen_x, sixteen_y), (zero_x, zero_y))
	
	dis_right = dist.euclidean((eye_right_x, eye_right_y), (brow_right_x, brow_right_y))
	#print "dis right " + str(dis_right)
	dis_left = dist.euclidean((eye_left_x, eye_left_y), (brow_left_x, brow_left_y))
	#print "dis left " + str(dis_left)
	
	return face_width, dis_right, dis_left

def estimate_distance_mouth(sixteen_x, sixteen_y, zero_x, zero_y, mouth_down_x, mouth_down_y, mouth_up_x, mouth_up_y):
	face_width = dist.euclidean((sixteen_x, sixteen_y), (zero_x, zero_y))
	
	dis = dist.euclidean((mouth_down_x, mouth_down_y), (mouth_up_x, mouth_up_y))
	#print "dis " + str(dis)
	
	return face_width, dis
	
def modify_pattern_brows(i, fil, zero_x, zero_y, sixteen_x, sixteen_y, ratio):
	# rotate
	w, h = fil.size
	width.append(w)
	height.append(h)
	angle = np.rad2deg(np.arctan2(zero_y - sixteen_y, sixteen_x - zero_x))
	ang.append(np.arctan2(zero_y - sixteen_y, sixteen_x - zero_x))
	rotated = fil.rotate(angle, Image.NEAREST, True)
	rotated.save("resized_%d.png" % i)
	
	# resize
	rotated = cv2.imread("resized_%d.png" % i, cv2.IMREAD_UNCHANGED)
	computed_r = float(ratio) / float(rotated.shape[1])
	dim = (ratio+90, int(rotated.shape[0] * computed_r)+90)
	res = cv2.resize(rotated, dim, interpolation = cv2.INTER_AREA)
	cv2.imwrite("resized_%d.png" % i, res)
	
def modify_pattern_mouth(i, fil, zero_x, zero_y, sixteen_x, sixteen_y, ratio):
	# rotate
	w, h = fil.size
	width.append(w)
	height.append(h)
	angle = np.rad2deg(np.arctan2(zero_y - sixteen_y, sixteen_x - zero_x))
	ang.append(np.arctan2(zero_y - sixteen_y, sixteen_x - zero_x))
	rotated = fil.rotate(angle, Image.NEAREST, True)
	rotated.save("resized_%d.png" % i)
	
	# resize
	rotated = cv2.imread("resized_%d.png" % i, cv2.IMREAD_UNCHANGED)
	computed_r = float(ratio) / float(rotated.shape[1])
	dim = (ratio, int(rotated.shape[0] * computed_r))
	res = cv2.resize(rotated, dim, interpolation = cv2.INTER_AREA)
	cv2.imwrite("resized_%d.png" % i, res)

def filter_image(f, n):
	sixteen_x = []
	sixteen_y = []
	zero_x = []
	zero_y = []
	nose_x = []
	nose_y = []
	brow_left_x = []
	brow_left_y = []
	brow_right_x = []
	brow_right_y = []
	eye_left_x = []
	eye_left_y = []
	eye_right_x = []
	eye_right_y = []
	border_mouth_upx = []
	border_mouth_upy = []
	border_mouth_downx = []
	border_mouth_downy = []
	inner_eye_right_x = []
	inner_eye_right_y = []
	inner_eye_left_x = []
	inner_eye_left_y = []	
	ratio = []
	
	# Read the image  
	image = cv2.imread(f)  
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	   
	# Detect faces in the image  
	faces = faceCascade.detectMultiScale(  
	  gray,  
	  scaleFactor=1.05,  
	  minNeighbors=5,  
	  minSize=(100, 100),  
	  flags=cv2.CASCADE_SCALE_IMAGE  
	)
	   
	# Draw a rectangle around the faces  
	for (x, y, w, h) in faces:  
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
	   
	   	# Converting the OpenCV rectangle coordinates to Dlib rectangle  
	   	dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))  
	   
	   	landmarks = np.matrix([[p.x, p.y]  
			for p in predictor(image, dlib_rect).parts()])  
	   
	   	for idx, point in enumerate(landmarks):  
	     		pos = (point[0, 0], point[0, 1])  
	     		cv2.putText(image, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))  
	     		cv2.circle(image, pos, 2, color=(0, 255, 255), thickness=-1)
	     		
	     		if idx == 0:
	     			zero_x.append(pos[0])
	     			zero_y.append(pos[1])
	     		elif idx == 16:
	     			sixteen_x.append(pos[0])
	     			sixteen_y.append(pos[1])
	     		elif idx == 33:
	     			nose_x.append(pos[0])
	     			nose_y.append(pos[1])
	     		elif idx == 19:
	     			brow_right_x.append(pos[0])
	     			brow_right_y.append(pos[1])
	     		elif idx == 24:
	     			brow_left_x.append(pos[0])
	     			brow_left_y.append(pos[1])
	     		elif idx == 37:
	     			eye_right_x.append(pos[0])
	     			eye_right_y.append(pos[1])
	     		elif idx == 44:
	     			eye_left_x.append(pos[0])
	     			eye_left_y.append(pos[1])
	     		elif idx == 62:
	     			border_mouth_upx.append(pos[0])
	     			border_mouth_upy.append(pos[1])
	     		elif idx == 66:
	     			border_mouth_downx.append(pos[0])
	     			border_mouth_downy.append(pos[1])
	     		elif idx == 39:
	     			inner_eye_right_x.append(pos[0])
	     			inner_eye_right_y.append(pos[1])
	     		elif idx == 42:
	     			inner_eye_left_x.append(pos[0])
	     			inner_eye_left_y.append(pos[1])

	for i in range(len(zero_x)):
		r = sixteen_x[i] - zero_x[i]
		ratio.append(r)
	
	if n == "p" or n == "k":
		for i in range(len(brow_right_x)):
			face_width, dis_right, dis_left = estimate_distance_brows(sixteen_x[i], sixteen_y[i], zero_x[i], zero_y[i], eye_right_x[i], eye_right_y[i], brow_right_x[i], brow_right_y[i], eye_left_x[i], eye_left_y[i], brow_left_x[i], brow_left_y[i])
			
			fil = open_brows_filter(face_width, dis_right, dis_left, n)
	elif n == "e":
		for i in range(len(border_mouth_upx)):
			face_width, dis = estimate_distance_mouth(sixteen_x[i], sixteen_y[i], zero_x[i], zero_y[i], border_mouth_downx[i], border_mouth_downy[i], border_mouth_upx[i], border_mouth_upy[i])
			
			fil = open_mouth_filter(face_width, dis, n)

	for i in range(len(ratio)):
		if n == "p" or n == "k":
			modify_pattern_brows(i, fil, zero_x[i], zero_y[i], sixteen_x[i], sixteen_y[i], ratio[i])
		elif n == "e":
			modify_pattern_mouth(i, fil, zero_x[i], zero_y[i], sixteen_x[i], sixteen_y[i], ratio[i])
		
	original = Image.open(f)

	for i in range(len(ratio)):
		if n == "p" or n == "k":
			paste_pattern_brows(n, i, original, ang[i], width[i], height[i], nose_x[i], nose_y[i])
		elif n == "e":
			paste_pattern_mouth(n, i, original, ang[i], width[i], height[i], inner_eye_right_x[i], inner_eye_right_y[i], inner_eye_left_x[i], inner_eye_left_y[i])
		
	for i in range(len(ratio)):
		os.remove("resized_%d.png" % i)
		

if __name__ == "__main__":
	global fil
	
	while success:
		cv2.imwrite("frame_%d.png" % count, frame)
		success, frame = vidcap.read()
		count += 1
	
	file_list = glob.glob('*.png')

	list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.png')[0]))

	for f in file_list:
		filter_image(f, filterChoice)

	with open('image_list.txt', 'w') as file:
	    for item in file_list:
		file.write("%s\n" % item)

	if filterChoice == "p":
		os.system('convert @image_list.txt {}.gif'.format("new/" + str(gif_name) + "_panda"))
	elif filterChoice == "k":
		os.system('convert @image_list.txt {}.gif'.format("new/" + str(gif_name) + "_pikachu"))
	elif filterChoice == "e":
		os.system('convert @image_list.txt {}.gif'.format("new/" + str(gif_name) + "_eyes"))

	for i in range(count):
		os.remove("frame_%d.png" % i)
		
	os.remove("image_list.txt")
	os.remove("outpy.avi")
