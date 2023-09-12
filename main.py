import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import cv2
import numpy as np

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

import cv2


def list_ports():
    """
    Test the ports and returns a tuple with the available ports
    and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports


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

ret, frame = cap.read()
frame = cv2.flip(frame, 1)
frame = frame[..., ::-1].copy()
img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
recognition_result = gesture_recognizer.recognize(img)
detection_result = pose_recognizer.detect(img)

hand_info = {
    "left": {},
    "right": {}
}
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = frame[..., ::-1].copy()
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = gesture_recognizer.recognize(img)
    if cnt % pose_freq == 0:
        detection_result = pose_recognizer.detect(img)

    annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
    annotated_image = annotated_image[..., ::-1].copy()
    cv2.imshow('frame', annotated_image)
    #segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    #visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    #cv2.imshow('mask', visualized_mask)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()