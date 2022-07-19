import cv2
from random import randrange

# load some pre_trained data on faces' frontal from opencv
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# initialize the web cam fpr real time detection
webcam = cv2.VideoCapture(0)

while True:
    # Read the current frame, the first variable is bool, the second one is the frame itself
    successful_frame_read, frame = webcam.read()

    # this one to make the application better by converting the image to greyed_color
    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grey_img)

    # draw the rectangle
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    # the first parameter of the function is the title of the window that appears
    cv2.imshow("Clever Programmer Face Detector", frame)

    # this one to make the image visible until the user press a key.
    # if you entered a letter, it will refer to a button in the keyboard
    # if you entered a number, it will refer to time in milliseconds
    key = cv2.waitKey(1)

    # stop if Q or q key is pressed
    if key == 81 or key == 113:
        break

# release the webcam object
webcam.release()
