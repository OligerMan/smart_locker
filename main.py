import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import cv2
import numpy as np
import time
import serial
from enum import Enum
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)

from pynput.keyboard import Controller, Key

kb = Controller()

class command(Enum):
    NO_DATA = -1
    LEFT = 0
    RIGHT = 1

gesture_model_path = 'models/gesture_recognizer.task'
pose_model_path = 'models/pose_landmarker.task'

gesture_base_options = python.BaseOptions(model_asset_path=gesture_model_path)
gesture_options = vision.GestureRecognizerOptions(base_options=gesture_base_options)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    output_segmentation_masks=True)
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
pose_freq = 5
last_trigger = time.time()

ret, frame = cap.read()
frame = cv2.flip(frame, 1)
frame = frame[..., ::-1].copy()
img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
recognition_result = gesture_recognizer.recognize(img)
detection_result = pose_recognizer.detect(img)

while True:
    hand_data = {
        "Left": {},
        "Right": {}
    }
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = frame[..., ::-1].copy()
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = gesture_recognizer.recognize(img)
    if cnt % pose_freq == 0:
        detection_result = pose_recognizer.detect(img)
    for gesture, hand_info in zip(recognition_result.gestures, recognition_result.handedness):
        gesture_info = set([gesture[0].category_name])
        hand_data[hand_info[0].category_name] = gesture_info
    if "Thumb_Up" in hand_data["Left"] or "Thumb_Up" in hand_data["Right"]:
        if time.time() - last_trigger > 1.5:
            kb.press(Key.right)
            kb.release(Key.right)
            last_trigger = time.time()
            print("right")
            arduino.write(command.RIGHT.value.to_bytes(1, byteorder='big', signed=True))
    if "Thumb_Down" in hand_data["Left"] or "Thumb_Down" in hand_data["Right"]:
        if time.time() - last_trigger > 1.5:
            kb.press(Key.left)
            kb.release(Key.left)
            last_trigger = time.time()
            print("left")
            arduino.write(command.LEFT.value.to_bytes(1, byteorder='big', signed=True))

    annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
    annotated_image = annotated_image[..., ::-1].copy()
    cv2.imshow('frame', annotated_image)
    #segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    #visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    #cv2.imshow('mask', visualized_mask)
    data = arduino.readline()
    print("Data from arduino:", data)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()