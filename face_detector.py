import cv2 as cv

from random import randrange

# Getting pre-trained data from open-cv to detect faces (haarcascade)
trained_face_data = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read video
webcam = cv.VideoCapture("vid_test.3gp")

while True:
    # Read currunt frame
    successful_frame_read, frame = webcam.read()

    # Make Video grayscale
    grayscaled_vid = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Getting coordinates of face
    face_coordinates = trained_face_data.detectMultiScale(frame)

    # Draw rectangle around face coordinates
    for (x, y, w, h) in face_coordinates:
        cv.rectangle(frame, (x, y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 10)

    # Show vid
    cv.imshow("Face Decetor app", frame)
    cv.waitKey(150)
