# gets images from the camera and saves them in the dataset folder

import cv2
import numpy as np
import os
CAMERAIP = "192.168.1.12/record"
choice = input("""Enter the type of image you want to capture:
  1. train
  2. test
""")

Itype = 'train'

if choice == '1':
    Itype = 'train'
elif choice == '2':
    Itype = 'test'

# Hardcoded className variable
className = input("Enter the class name: ")

# Directory to store images
output_dir = os.path.join(os.getcwd(), "dataset", Itype, className)

# Check if the directory exists; if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to count existing images in the directory


def preprocess_image(image):
    """Resize and normalize the image to 32x32."""
    target_size = (32, 32)  # Change the target size to 32x32
    image = cv2.resize(image, target_size)  # Resize to model's input size
    return image


def count_images_in_directory(directory):
    return len([file for file in os.listdir(directory) if file.endswith('.jpg')])


# Open video stream (0 for the default camera)
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Loop to read images from the stream
try:
    for i in range(1000):  # Change the number of images to capture
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from stream.")
            break

        # Display the frame (optional)
        cv2.imshow('Video Stream', frame)

        # Count existing images and determine the next image number
        next_image_number = count_images_in_directory(output_dir)
        image_filename = f"{className}{next_image_number}.jpg"
        image_path = os.path.join(output_dir, image_filename)

        # Save the current frame as an image
        cv2.imwrite(image_path, preprocess_image(frame))
        print(f"Image saved at: {image_path}")

        # Wait for a key press; break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release the video stream and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
