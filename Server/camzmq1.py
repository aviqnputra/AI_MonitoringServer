import zmq
import numpy as np
import time
import cv2
import os

context = zmq.Context()

image_socket = context.socket(zmq.SUB)
image_socket.bind("tcp://127.0.0.1:8005")
image_socket.setsockopt_string(zmq.SUBSCRIBE, '')
image_dir = "/root/projectAI/Yolo/Input"

os.makedirs(image_dir, exist_ok=True)
while True:
	image_bytes = image_socket.recv()
	image_array = np.frombuffer(image_bytes, dtype=np.uint8)
	image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
	image_path = os.path.join(image_dir, f"image_{int(time.time())}.jpg")
	cv2.imwrite(image_path, image)
	print(f"image save: {image_path}")
