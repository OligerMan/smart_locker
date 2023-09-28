import math
import os
import shlex
import subprocess

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
import glob
import multiprocessing

from mediapipe.tasks.python.vision import RunningMode

from arduino_control import Command, Arduino
from pynput.keyboard import Controller, Key
import face_recognition

face_encodings = []
face_encodings_tags = []
for path in glob.glob("faces/*"):
    img = face_recognition.load_image_file(path)
    enc = face_recognition.face_encodings(img)[0]
    face_encodings.append(enc)
    face_encodings_tags.append(path[6:-4])


def faces_detection(img_queue, res_queue):
    print("Face detection service started")
    while True:
        img = img_queue.get()
        while not img_queue.empty():
            img = img_queue.get()
        #print("got new image")
        out = [False for _ in face_encodings]
        for face_encoding in face_recognition.face_encodings(img):
            detection = face_recognition.compare_faces(face_encodings, face_encoding, tolerance=0.4)
            out = [elem1 or elem2 for elem1, elem2 in zip(out, detection)]
        res_queue.put(out)


def open_cell(arduino, tags, permissions, target):
    for tag in tags:
        if permissions.get(tag, None) is not None:
            if target in permissions[tag]:
                arduino.send_command(Command.OPEN_CELL)
                print("Cell opened")
                return
    arduino.send_command(Command.OPEN_CELL_NO_PERMISSION)
    print("No permission")


def close_cell(arduino, tags, permissions, target):
    for tag in tags:
        if permissions.get(tag, None) is not None:
            if target in permissions[tag]:
                arduino.send_command(Command.CLOSE_CELL)
                print("Cell closed")
                return
    arduino.send_command(Command.CLOSE_CELL_NO_PERMISSION)
    print("No permission")


if __name__ == '__main__':
    current_cell_selection = "left_up"
    permissions = {
        "oleg": ["left_down", "right_up", "right_down"],
        "luca": ["left_up", "left_down", "right_up", "right_down"],
        "maksim": ["left_up", "left_down", "right_up", "right_down"],
        "nikita": ["left_up", "left_down", "right_up", "right_down"],
        "vika": ["left_up", "left_down", "right_up", "right_down"]
    }

    img_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()

    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=faces_detection, args=(img_queue, res_queue))
    p.start()

    kb = Controller()

    gesture_model_path = 'models/gesture_recognizer.task'
    pose_model_path = 'models/pose_landmarker.task'
    face_model_path = 'models/blaze_face_short_range.tflite'

    gesture_base_options = python.BaseOptions(model_asset_path=gesture_model_path, delegate=1)
    gesture_options = vision.GestureRecognizerOptions(
        base_options=gesture_base_options,
        num_hands=10)
    gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

    pose_base_options = python.BaseOptions(model_asset_path=pose_model_path, delegate=1)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        output_segmentation_masks=False,
        num_poses=2)
    pose_recognizer = vision.PoseLandmarker.create_from_options(pose_options)

    face_base_options = python.BaseOptions(model_asset_path=face_model_path, delegate=1)
    face_options = vision.FaceDetectorOptions(
        base_options=face_base_options)
    face_recognizer = vision.FaceDetector.create_from_options(face_options)


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
    pose_freq = 1
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
        ("Left", "Open_Palm"): open_cell,
        ("Right", "Open_Palm"): open_cell,
        ("Left", "Closed_Fist"): close_cell,
        ("Right", "Closed_Fist"): close_cell
    }

    gesture_trigger_delay = 1
    try:
        arduino = Arduino('COM3')
    except Exception:
        arduino = Arduino()
        print("failed arduino start")

    face_detection_window_size = 7
    prev_detection_window = [set() for i in range(face_detection_window_size)]
    face_permission_tags = set()

    while True:
        cnt += 1
        start_point = time.time()
        hand_data = {
            "Left": {},
            "Right": {}
        }
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        scale_percent = 25  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        frame = frame[..., ::-1].copy()
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        resized_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)
        gesture_recognition_result = gesture_recognizer.recognize(img)
        """face_recognition_result = face_recognizer.detect(img)
        for res in face_recognition_result.detections:
            pass
            #print(res.categories)"""
        if cnt % pose_freq == 0:
            detection_result = pose_recognizer.detect(resized_img)
        for gesture, hand_info in zip(gesture_recognition_result.gestures, gesture_recognition_result.handedness):
            gesture_info = set([gesture[0].category_name])
            hand_data[hand_info[0].category_name] = gesture_info
        if img_queue.empty():
            img_queue.put(frame)
        try:
            data = res_queue.get_nowait()
            prev_detection = set()
            for detection in prev_detection_window:
                prev_detection = prev_detection.union(detection)
            prev_detection_window = [tag_set for i, tag_set in enumerate(prev_detection_window) if i != 0]
            current_detection = set()

            for i, tag in enumerate(face_encodings_tags):
                if data[i]:
                    #print(tag, "detected, frame", cnt)
                    current_detection.update([tag])

            prev_detection_window.append(current_detection)
            current_detection = set()
            for detection in prev_detection_window:
                current_detection = current_detection.union(detection)
            face_permission_tags = current_detection
            if prev_detection != current_detection:
                # print(prev_detection, current_detection)
                for tag in list(prev_detection.difference(current_detection)):
                    print(tag, "out of frame")
                for tag in list(current_detection.difference(prev_detection)):
                    print(tag, "now in frame")
                    command = "vlc voices/" + tag + ".mp3"
                    proc = subprocess.Popen(
                        shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
                    )
        except Exception:
            pass
        if time.time() - last_trigger > gesture_trigger_delay:
            for data in hand_data["Right"]:
                print(data)
                gesture_mapping.get(("Right", data), stub)(arduino, face_permission_tags, permissions, current_cell_selection)
                last_trigger = time.time()
            for data in hand_data["Left"]:
                print(data)
                gesture_mapping.get(("Left", data), stub)(arduino, face_permission_tags, permissions, current_cell_selection)
                last_trigger = time.time()
        if len(detection_result.pose_landmarks):
            elbow = detection_result.pose_landmarks[0][14]
            wrist = detection_result.pose_landmarks[0][16]
            diff_x = math.atan((elbow.x - wrist.x) * 1000)
            diff_y = math.atan((elbow.y - wrist.y) * 1000)
            if elbow.presence > 0.75 and wrist.presence > 0.75:
                if diff_x < 0 and diff_y > 0:
                    arduino.send_command(Command.SELECT_RIGHT_UP)
                    current_cell_selection = "right_up"
                    #print("right_up")
                if diff_x < 0 and diff_y < 0:
                    arduino.send_command(Command.SELECT_RIGHT_DOWN)
                    current_cell_selection = "right_down"
                    #print("right_down")
                if diff_x > 0 and diff_y > 0:
                    arduino.send_command(Command.SELECT_LEFT_UP)
                    current_cell_selection = "left_up"
                    #print("left_up")
                if diff_x > 0 and diff_y < 0:
                    arduino.send_command(Command.SELECT_LEFT_DOWN)
                    current_cell_selection = "left_down"
                    #print("left_down")
        #print(time.time() - start_point)
        annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
        annotated_image = annotated_image[..., ::-1].copy()
        cv2.imshow('frame', annotated_image)
        #print(arduino.get_data())
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()