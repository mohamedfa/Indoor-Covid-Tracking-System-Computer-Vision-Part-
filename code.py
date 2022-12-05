import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
from time import sleep
import RPi.GPIO as GPIO
import smbus
import sys
import time

# Define some device parameters
I2C_ADDR  = 0x27 # I2C device address
LCD_WIDTH = 16   # Maximum characters per line

# Define some device constants
LCD_CHR = 1 # Mode - Sending data
LCD_CMD = 0 # Mode - Sending command

LCD_LINE_1 = 0x80 # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0 # LCD RAM address for the 2nd line
LCD_LINE_3 = 0x94 # LCD RAM address for the 3rd line
LCD_LINE_4 = 0xD4 # LCD RAM address for the 4th line

LCD_BACKLIGHT  = 0x08  # On
#LCD_BACKLIGHT = 0x00  # Off

ENABLE = 0b00000100 # Enable bit

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005

#Open I2C interface
#bus = smbus.SMBus(0)  # Rev 1 Pi uses 0
bus = smbus.SMBus(1) # Rev 2 Pi uses 1

def lcd_init():
  # Initialise display
  lcd_byte(0x33,LCD_CMD) # 110011 Initialise
  lcd_byte(0x32,LCD_CMD) # 110010 Initialise
  lcd_byte(0x06,LCD_CMD) # 000110 Cursor move direction
  lcd_byte(0x0C,LCD_CMD) # 001100 Display On,Cursor Off, Blink Off 
  lcd_byte(0x28,LCD_CMD) # 101000 Data length, number of lines, font size
  lcd_byte(0x01,LCD_CMD) # 000001 Clear display
  time.sleep(E_DELAY)

def lcd_byte(bits, mode):
  # Send byte to data pins
  # bits = the data
  # mode = 1 for data
  #        0 for command

  bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT
  bits_low = mode | ((bits<<4) & 0xF0) | LCD_BACKLIGHT

  # High bits
  bus.write_byte(I2C_ADDR, bits_high)
  lcd_toggle_enable(bits_high)

  # Low bits
  bus.write_byte(I2C_ADDR, bits_low)
  lcd_toggle_enable(bits_low)

def lcd_toggle_enable(bits):
  # Toggle enable
  time.sleep(E_DELAY)
  bus.write_byte(I2C_ADDR, (bits | ENABLE))
  time.sleep(E_PULSE)
  bus.write_byte(I2C_ADDR,(bits & ~ENABLE))
  time.sleep(E_DELAY)

def lcd_string(message,line):
  # Send string to display

  message = message.ljust(LCD_WIDTH," ")

  lcd_byte(line, LCD_CMD)

  for i in range(LCD_WIDTH):
    lcd_byte(ord(message[i]),LCD_CHR)

relay = 18
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay, GPIO.OUT)
GPIO.output(relay ,0)

#lcd_init()

def faceRecognition():
	now = datetime.now()
	time_hour = now.strftime('%H')

	pplImages = []
	pplNames = []

	path = '/home/pi/project/Merge/Resources'
	myList = os.listdir(path)
	print(myList)
	for cl in myList:
		img = cv2.imread(f'{path}/{cl}')
		pplImages.append(img)
		pplNames.append(os.path.splitext(cl)[0])
	print(pplNames)

	def findEncodings(images):
		encodeList = []
		for img in images:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			encode = face_recognition.face_encodings(img)[0]
			encodeList.append(encode)
		return encodeList

	encodeListKnown = findEncodings(pplImages)
	print('Encoding Complete')

	MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

	def detect_and_predict_age(frame, faceNet, ageNet, genderNet):
		AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
		GENDER_BUCKETS = ["Male", "Female"]
		results = []
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
		faceNet.setInput(blob)
		detections = faceNet.forward()

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY:endY, startX:endX]
				if face.shape[0] < 20 or face.shape[1] < 20:
					continue
				faceBlob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
				global j
				j = i
				ageNet.setInput(faceBlob)
				predsA = ageNet.forward()
				i = predsA[0].argmax()
				age = AGE_BUCKETS[i]
				ageConfidence = predsA[0][i]
				genderNet.setInput(faceBlob)
				predsG = genderNet.forward()
				j = predsG[0].argmax()
				gender = GENDER_BUCKETS[j]
				genderConfidence = predsG[0][j]
				d = {"loc": (startX, startY, endX, endY), "age": (age, ageConfidence), "gender": (gender, genderConfidence)}
				results.append(d)
		return results

	print("[INFO] Loading face detector model...")
	prototxtPath = r"/home/pi/project/Merge/face_detector/deploy.prototxt"
	weightsPath = r"/home/pi/project/Merge/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
	print("[INFO] Loading age detector model...")
	prototxtPath = r"/home/pi/project/Merge/age_detector/age_deploy.prototxt"
	weightsPath = r"/home/pi/project/Merge/age_detector/age_net.caffemodel"
	ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)
	print("[INFO] Loading gender detector model..")
	prototxtPath = r"/home/pi/project/Merge/gender_detector/gender.prototxt"
	weightsPath = r"/home/pi/project/Merge/gender_detector/gender_net.caffemodel"
	genderNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	#camera_num = 0
	#cam = cv2.VideoCapture(camera_num)
	vs = VideoStream(src=0).start()
	#lcd_string("Unwear Your Mask",LCD_LINE_1)
    #lcd_string("Look at the Cam",LCD_LINE_2)

	while True:
	# success, frame = cam.read()
		frame = vs.read()
		frame = imutils.resize(frame, width=800)
		imgX = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
		imgX = cv2.cvtColor(imgX, cv2.COLOR_BGR2RGB)
    
		results = detect_and_predict_age(imgX, faceNet, ageNet, genderNet)
		'''
		for r in results:
        text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100) + "  " + "{}: {:.2f}%".format(r["gender"][0], r["gender"][1] * 100)
        (startX, startY, endX, endY) = r["loc"]
        y = startY - 20 if startY - 20 > 20 else startY + 20
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 3)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (205, 205, 0), 2)
        '''
        

		facesCurFrame = face_recognition.face_locations(imgX)
		#print(facesCurFrame)
    
		encodesCurFrame = face_recognition.face_encodings(imgX, facesCurFrame)

		for encodeFace, faceLoc, result in zip(encodesCurFrame, facesCurFrame, results):
			y1, x2, y2, x1 = faceLoc
			matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
			faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
			text = "{}: {:.2f}%".format(result["age"][0], result["age"][1] * 100) + "  " + "{}: {:.2f}%".format(result["gender"][0], result["gender"][1] * 100)
			#(startX, startY, endX, endY) = result["loc"]
			#y = startY - 20 if startY - 20 > 20 else startY + 20
			#cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 3)
			cv2.putText(frame, text, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


			matchIndex = np.argmin(faceDis)

			room_user=[]
			if matches[matchIndex]:
				name = pplNames[matchIndex].capitalize()
				
				return name

				for nm in room_user:
					if name != nm:
						room_user.append(name)
						#sendData(matchIndex, camera_num, name, session_num)
                    
						sql_query = "insert into room_user(user_iduser, room_idroom, name, session_num) values(%s,%s,%s,%s)"
						values = (2, camera_num, name, time_hour)
						#mycursor.execute(sql_query, values)
						#db_address.commit()

            
				y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
				cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
				cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
				#markAttendance(name)
			else:
				y1, x2, y2, x1 = faceLoc
				y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
				cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
				cv2.putText(frame, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

		cv2.imshow('WebCam', frame)
		k = cv2.waitKey(1)

		# press 'ESC' to quit
		if k == 27:
			break

	vs.stop()
	cv2.destroyAllWindows()



def maskDetector():
    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()
        #print(detections.shape)

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)
    # load our serialized face detector model from disk
    prototxtPath = r"/home/pi/project/face_detector/deploy.prototxt"
    weightsPath = r"/home/pi/project/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model(r"/home/pi/project/mask_detector.modelll")

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            if label == "Mask":
                GPIO.output(relay, 1)
                lcd_string("Thanks, Now you ",LCD_LINE_1)
                lcd_string("can enter :)",LCD_LINE_2)
                sleep(5)
                GPIO.output(relay ,0)
                lcd_init()
                
                return print("Thanks, Now you can enter.")

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


faceRecognition()
#lcd_init()
sleep(2)
#lcd_string("Wear Your Mask",LCD_LINE_1)
#lcd_string(":)",LCD_LINE_2)
print("Wear Your Mask")

maskDetector()

