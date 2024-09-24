

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('test.h5')

# Define the target size that the model expects
target_size = (32, 32)  # Adjust according to your model's input size


def capture_image():
    # Open the camera (0 for the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    print("Press 's' to take a photo.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('Camera', frame)  # Show the camera feed

        key = cv2.waitKey(1)
        if key == ord('s'):  # Capture the image when 's' is pressed
            captured_image = frame
            break

    cap.release()
    cv2.destroyAllWindows()

    return captured_image


def preprocess_image(image):
    """Resize and normalize the image."""
    image = cv2.resize(image, target_size)  # Resize to model's input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def predict_image(image):
    """Make a prediction on the given image."""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    print("Predictions:", predictions)
    return predicted_class


# Main function to capture an image and predict its class
if __name__ == "__main__":
    image = capture_image()
    if image is not None:
        predicted_class = predict_image(image)
        print("Predicted class:", predicted_class)
