import math

import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import cv2
import numpy as np
import time
import serial
from enum import Enum

from arduino_control import Command, Arduino

from pynput.keyboard import Controller, Key

kb = Controller()


gesture_model_path = 'models/gesture_recognizer.task'
pose_model_path = 'models/pose_landmarker.task'

gesture_base_options = python.BaseOptions(model_asset_path=gesture_model_path)
gesture_options = vision.GestureRecognizerOptions(base_options=gesture_base_options)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    output_segmentation_masks=False)
pose_recognizer = vision.PoseLandmarker.create_from_options(pose_options)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    print(pose_landmarks_proto)
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cnt = 0
pose_freq = 50000
last_trigger = time.time()

ret, frame = cap.read()
frame = cv2.flip(frame, 1)
frame = frame[..., ::-1].copy()
img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
recognition_result = gesture_recognizer.recognize(img)
detection_result = pose_recognizer.detect(img)

def stub(*args, **kwargs):
    pass

gesture_mapping = {
    ("Left", "Thumb_Up"): lambda arduino: arduino.send_command(Command.LEFT_UP),
    ("Right", "Thumb_Up"): lambda arduino: arduino.send_command(Command.RIGHT_UP),
    ("Left", "Thumb_Down"): lambda arduino: arduino.send_command(Command.LEFT_DOWN),
    ("Right", "Thumb_Down"): lambda arduino: arduino.send_command(Command.RIGHT_DOWN)
}


gesture_trigger_delay = 1
try:
    arduino = Arduino('COM3')
except Exception:
    arduino = Arduino()
    print("failed arduino start")

pose_angle_data = {
    "x": 0,
    "y": 0
}

while True:
    cnt += 1
    start_point = time.time()
    hand_data = {
        "Left": {},
        "Right": {}
    }
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    frame = frame[..., ::-1].copy()
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    resized_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)
    start_point = time.time()
    recognition_result = gesture_recognizer.recognize(img)
    if cnt % pose_freq == 0:
        detection_result = pose_recognizer.detect(resized_img)
    for gesture, hand_info in zip(recognition_result.gestures, recognition_result.handedness):
        gesture_info = set([gesture[0].category_name])
        hand_data[hand_info[0].category_name] = gesture_info
    if time.time() - last_trigger > gesture_trigger_delay:
        for data in hand_data["Right"]:
            print(data)
            gesture_mapping.get(("Right", data), stub)(arduino)
            last_trigger = time.time()
        for data in hand_data["Left"]:
            print(data)
            gesture_mapping.get(("Left", data), stub)(arduino)
            last_trigger = time.time()
    if len(detection_result.pose_landmarks):
        elbow = detection_result.pose_landmarks[0][13]
        wrist = detection_result.pose_landmarks[0][15]
        elbow_fix = landmark_pb2.NormalizedLandmark(x=elbow.x, y=elbow.y, z=elbow.z)
        wrist_fix = landmark_pb2.NormalizedLandmark(x=wrist.x, y=wrist.y, z=wrist.z)
        print("------")
        elbow_pixel = _normalized_to_pixel_coordinates(elbow_fix.x, elbow_fix.y, 640, 480)
        elbow_pixel = _normalized_to_pixel_coordinates(elbow_fix.x, elbow_fix.y, 640, 480)
    #print(time.time() - start_point)
    annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
    annotated_image = annotated_image[..., ::-1].copy()
    cv2.imshow('frame', annotated_image)
    print(arduino.get_data())
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()