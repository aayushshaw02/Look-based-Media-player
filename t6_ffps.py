import cv2 as cv
import numpy as np
import threading
import time
from ffpyplayer.player import MediaPlayer
# Load face cascade and eye cascade from haarcascades folder
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

# Initialize video player state
playing = True
paused = False
audio_paused=False
lock = threading.Lock()
# player = MediaPlayer(video_file)
# Define a function to handle video playback
def play_video(video_file):
    global playing, paused, lock , audio_paused
    
    # Open the video file for playing
    video_player = cv.VideoCapture(video_file)
    player = MediaPlayer(video_file)
    last_video_timestamp=0
    
    while True:
        lock.acquire()  # Acquire the lock to prevent race conditions
        
        # If paused, wait for the 'p' key to be pressed to resume playing
        if paused:
            lock.release()
            time.sleep(0.01)  # Sleep briefly to avoid busy waiting
            continue
        
        # Read a frame from the video player
        ret_player, frame_player = video_player.read()
        audio_frame, val = player.get_frame()
        # If video player reaches the end of the video, start playing it again from the beginning
        if not ret_player:
            video_player.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # video_timestamp = video_player.get(cv.CAP_PROP_POS_MSEC)
        # audio_timestamp = player.get_pts() * 1000.0
        video_timestamp = video_player.get(cv.CAP_PROP_POS_MSEC)
        timestamp_diff = video_timestamp - last_video_timestamp
        last_video_timestamp = video_timestamp
        
        # Compute the difference between the timestamps
        # timestamp_diff = audio_timestamp - video_timestamp
        
        # If the audio is ahead of the video, wait for the video to catch up
        # if timestamp_diff > 5:
        #     time.sleep(timestamp_diff / 1000.0)
        # Adjust video playback speed based on timestamp difference
        if timestamp_diff > 0:
            play_speed = 1.0 + (timestamp_diff / 1000.0)
            video_player.set(cv.CAP_PROP_POS_FRAMES, video_player.get(cv.CAP_PROP_POS_FRAMES) + int(play_speed))


        # Display the next frame
        cv.imshow('Video', frame_player)
        # cv.namedWindow('Video', cv.WINDOW_NORMAL)
        # cv.namedWindow('Webcam', cv.WINDOW_NORMAL)
        # cv.resizeWindow('Video', 800, 600)
        # cv.resizeWindow('Webcam', 800, 600)

        # if not audio_paused:
        #     player.toggle_pause()
        # If not playing, wait for the 'p' key to be pressed to pause the video
        #player.toggle_pause()
        if not playing:
            lock.release()
            player.pause()
            # player.refresh()
            time.sleep(0.01)  # Sleep briefly to avoid busy waiting
            continue
        
        lock.release()  # Release the lock
        
        # Wait for a short time before displaying the next frame
        cv.waitKey(25)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame


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
        audio_paused = False
        paused = False
        lock.release()  # Release the lock

    # If no faces are detected, pause video playback
    else:
        lock.acquire()  # Acquire the lock to prevent race conditions
        playing = False
        
        audio_paused = True
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
