import keyboard
import cv2

# Initialize the video player and set the path to your video file
cap = cv2.VideoCapture('video.mp4')

# Set the initial state of the player to playing
playing = True

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame was successfully read, display it
    if ret:
        cv2.imshow('Video Player', frame)

        # Check if the 'p' key has been pressed to pause or play the video
        if keyboard.is_pressed('p'):
            playing = not playing
            print('Video Paused' if not playing else 'Video Playing')

        # If the video is playing, wait for a key press to display the next frame
        if playing:
            cv2.waitKey(25)

        # If the video is paused, wait for the 'p' key to be pressed again to resume playing
        else:
            while True:
                if keyboard.is_pressed('p'):
                    playing = True
                    print('Video Resumed')
                    break

    # If the frame could not be read, break the loop and exit the program
    else:
        break

# Release the video player and close all windows
cap.release()
cv2.destroyAllWindows()
