import cv2 as cv
import numpy as np
import threading
import time

# Load face cascade and eye cascade from haarcascades folder
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

# Initialize video player state
playing = True
paused = False
lock = threading.Lock()

# Define a function to handle video playback
def play_video(video_file):
    global playing, paused, lock
    
    # Open the video file for playing
    video_player = cv.VideoCapture(video_file)
    
    while True:
        lock.acquire()  # Acquire the lock to prevent race conditions
        
        # If paused, wait for the 'p' key to be pressed to resume playing
        if paused:
            lock.release()
            time.sleep(0.01)  # Sleep briefly to avoid busy waiting
            continue
        
        # Read a frame from the video player
        ret_player, frame_player = video_player.read()

        # If video player reaches the end of the video, start playing it again from the beginning
        if not ret_player:
            video_player.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        # Display the next frame
        cv.imshow('Video', frame_player)
        # cv.namedWindow('Video', cv.WINDOW_NORMAL)
        # cv.namedWindow('Webcam', cv.WINDOW_NORMAL)
        # cv.resizeWindow('Video', 800, 600)
        # cv.resizeWindow('Webcam', 800, 600)


        # If not playing, wait for the 'p' key to be pressed to pause the video
        if not playing:
            lock.release()
            time.sleep(0.01)  # Sleep briefly to avoid busy waiting
            continue
        
        lock.release()  # Release the lock
        
        # Wait for a short time before displaying the next frame
        cv.waitKey(25)

    video_player.release()

# Start the video playback thread
video_thread = threading.Thread(target=play_video, args=('video.mp4',))
video_thread.daemon = True
video_thread.start()

# Capture video from webcam
video_capture = cv.VideoCapture(0)

# Read all frames from webcam
while True:
    # Read a frame from the webcam
    ret_webcam, frame_webcam = video_capture.read()
    frame_webcam = cv.flip(frame_webcam, 1)  # Flip so that video feed is not flipped, and appears mirror-like.
    gray_webcam = cv.cvtColor(frame_webcam, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_webcam, 1.3, 5)

    # If faces are detected, resume video playback
    if len(faces) > 0:
        lock.acquire()  # Acquire the lock to prevent race conditions
        playing = True
        paused = False
        lock.release()  # Release the lock

    # If no faces are detected, pause video playback
    else:
        lock.acquire()  # Acquire the lock to prevent race conditions
        playing = False
        paused = False
        lock.release()  # Release the lock

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

    # Wait for a short time before reading the next frame
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
video_thread.stop()
video_capture.release()
cv.destroyAllWindows()
