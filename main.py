# import
import numpy as np
import cv2 as cv
import tensorflow.keras.models as models
from flask import Flask, jsonify
import requests
import time
global classes

classes = ["plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck", "none"]
global cameraIp
cameraIp = "192.168.1.12"
model = models.load_model('test.h5')


# transform image
def transform_image(image):
    image = cv.resize(image, (32, 32))
    image = image.reshape(1, 32, 32, 3)
    return image


app = Flask(__name__)

# Sample function to process the video frame


def wii(frame):
    prediction = model.predict(transform_image(frame))
    index = np.argmax(prediction)
    if max(prediction[0]) < 0.5:
        return "none"
    return classes[index]


# Function to continuously read video stream and send data
def continuously_process_and_send():
    video_url = f"{cameraIp}/record"

    # Open the video stream
    video_stream = cv.VideoCapture(video_url)

    if not video_stream.isOpened():
        print("Failed to open video stream")
        return

    last_result = "none"
    while True:
        try:
            # Read the frame from the video stream
            ret, frame = video_stream.read()

            if not ret:
                print("Failed to read frame. Retrying...")
                time.sleep(5)  # Wait before retrying
                continue  # Retry reading frame

            # Pass the frame to the wii function for processing
            result = wii(frame)
            if result != last_result:
                last_result = result
                # Prepare the hardcoded JSON response
                json_data = {
                    "product": result,
                    "status": "200"
                }

                # Send the result to the IP address
                requests.post('http://192.168.1.16', json=json_data)

#                if response.status_code == 200:
#                    print(f"yay data sent: {response.text}")
#                else:
#                    print(f"oh no the data was not sent: {
#                          response.status_code}, {response.text}")

            time.sleep(2)  # time between each read frame
        except requests.exceptions.RequestException as e:
            print(f"error with request: {e}")
            time.sleep(10)  # Wait before retrying to avoid rapid failures

        except Exception as e:
            print(f"error: {e}")
            video_stream.release()
            video_stream = cv.VideoCapture(video_url)  # Reopen the stream


@app.route('/', methods=['GET'])
def start_stream():
    # Start the continuous processing in a background thread or process
    import threading
    thread = threading.Thread(target=continuously_process_and_send)
    thread.start()
    return jsonify({"message": "Started processing and sending requests"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
