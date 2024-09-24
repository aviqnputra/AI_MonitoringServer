import zmq 
import cv2
import time
import numpy as np
context = zmq.Context()
image_socket = context.socket(zmq.PUB)
image_socket.connect("tcp://127.0.0.1:5005")
cap = cv2.VideoCapture(0)
running = True
def shutdown(signum, frame):
    global running
    running = False
import signal
signal.signal(signal.SIGINT, shutdown)
while running:
    ret, frame = cap.read()
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    image_socket.send(image_bytes)
    print("Image sent")
    time.sleep(2)
cap.release()
print("Camera released")
