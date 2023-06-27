import cv2
import numpy as np
import os
import sys
import time
import pygame
from keras.models import load_model
from pygame.locals import *
# # Save the model and optimizer state
# model.save('model.h5')
# model.save_weights('model_weights.h5')

# Load the model and optimizer state
model = load_model('model.h5')
# model.load_weights('model_weights.h5')
# Save the model and optimizer state
model.save('model.h5')
# model.save_weights('model_weights.h5')

# Load the pre-trained facial expression recognition model
model = load_model('model.h5')

# Define a list of emotion labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Define the predict_facial_expression() function
def predict_facial_expression(gray_image):
    # # Resize the image to match the input size of the model (48x48 pixels)
    # resized_image = cv2.resize(gray_image, (48, 48))
    
    # # Reshape the image to match the input shape of the model (1x48x48x1)
    # reshaped_image = resized_image.reshape(1, 48, 48, 1)
    
    # # Normalize the pixel values of the image to be between 0 and 1
    # normalized_image = reshaped_image / 255.0
    
    # # Use the model to make a prediction on the normalized image
    # predictions = model.predict(normalized_image)
    
    # # Get the index of the predicted emotion with the highest probability
    # predicted_index = np.argmax(predictions)
    
    # # Map the predicted index to an emotion label and return it
    # predicted_emotion = EMOTIONS[predicted_index]
    # return predicted_emotion

    # Resize the image to the expected size
    normalized_image = cv2.resize(roi_gray, (90, 3))
    normalized_image = normalized_image.reshape((1, 90, 3, 1))
    normalized_image = normalized_image.astype('float32') / 255.0
    
    # Make the prediction using the modified input image
    predictions = model.predict(normalized_image)
    return EMOTIONS[np.argmax(predictions)]

# Initialize the pygame library and set the window size
pygame.init()
pygame.display.set_caption("Video Player")
pygame.display.set_mode((640, 480))

# Load the video file
video_file = 'video.mp4'
cap = cv2.VideoCapture(video_file)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set up the initial state of the video player
playing = True
paused = False
rewinding = False
fast_forwarding = False
current_frame = 0

# Start playing the video
while cap.isOpened():
    # Read the current frame from the video
    ret, frame = cap.read()
    
    # If we have reached the end of the video, stop playing
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame using the face cascade classifier
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) corresponding to the face
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # Use the predict_facial_expression() function to get the current facial expression
        predicted_expression = predict_facial_expression(roi_gray)
        
        # Draw a rectangle around the detected face and display the current facial expression
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_expression, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    
#     # Convert the frame to grayscale
# gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # Convert the grayscale frame to BGR format
# frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)


    # Convert the frame back to BGR for display
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display the current frame in the pygame window
    surface = pygame.surfarray.make_surface(frame)
    window = pygame.display.get_surface()
    window.blit(surface, (0, 0))
    pygame.display.update()

    # Handle events from the user
    for event in pygame.event.get():
        if event.type == QUIT:
            # If the user clicks the 'X' button, stop playing the video
            playing = False
        elif event.type == KEYDOWN:
            if event.key == K_SPACE:
                # If the user presses the space bar, pause or resume the video
                paused = not paused
            elif event.key == K_LEFT:
                # If the user presses the left arrow key, rewind the video
                rewinding = True
            elif event.key == K_RIGHT:
                # If the user presses the right arrow key, fast-forward the video
                fast_forwarding = True

    # Update the current frame based on the player's state
    if not paused:
        if rewinding:
            current_frame -= 10
        elif fast_forwarding:
            current_frame += 10
        else:
            current_frame += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        rewinding = False
        fast_forwarding = False

# Release the video and clean up the pygame window
cap.release()
pygame.quit()
