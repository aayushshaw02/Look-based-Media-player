import cv2 as cv
import numpy as np
import keyboard
import os

# Load face cascade and eye cascade from haarcascades folder
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

# Capture video from webcam
video_capture = cv.VideoCapture(0)

# Read all frames from webcam
playing = True

# Open the video file for playing
file_path = 'video.mp4'
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File {file_path} not found")
video_player = cv.VideoCapture(file_path)

while True:
    # Read a frame from the video player
    ret_player, frame_player = video_player.read()

    # If video player reaches the end of the video, start playing it again from the beginning
    if not ret_player:
        video_player.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    # Read a frame from the webcam
    ret_webcam, frame_webcam = video_capture.read()
    frame_webcam = cv.flip(frame_webcam, 1)  # Flip so that video feed is not flipped, and appears mirror-like.
    gray_webcam = cv.cvtColor(frame_webcam, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_webcam, 1.3, 5)

    # If faces are detected, resume video playback
    if len(faces) > 0:
        playing = True

    # If no faces are detected, pause video playback
    else:
        playing = False

    # Display face rectangles and eye rectangles in the webcam feed
    for (x, y, w, h) in faces:
        cv.rectangle(frame_webcam, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_webcam[y:y + h, x:x + w]
        roi_color = frame_webcam[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the webcam feed
    cv.imshow('Webcam', frame_webcam)

    # If the video is playing, display the next frame
    if playing:
        cv.imshow('Video', frame_player)
        cv.waitKey(1)

    # If the video is paused, wait for the 'p' key to be pressed to resume playing
    else:
        while True:
            if keyboard.is_pressed('p'):
                playing = True
                print('Video Resumed')
                break

    # Check if the 'q' key has been pressed to exit the program
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
video_player.release()
cv.destroyAllWindows()
