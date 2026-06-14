#!/usr/bin/env python3
"""Simple face detection using OpenCV Haar cascades.

Supports webcam live detection and single-image annotation.
"""
import argparse
import sys
from pathlib import Path

import cv2


def detect_faces(image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    return faces


def draw_faces(image, faces, color=(0, 255, 0), thickness=2):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def process_image(input_path, output_path=None, show=True):
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: couldn't read image '{input_path}'")
        sys.exit(1)
    faces = detect_faces(image)
    draw_faces(image, faces)
    if output_path:
        cv2.imwrite(str(output_path), image)
    if show:
        cv2.imshow('Face Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"Detected {len(faces)} face(s)")


def process_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print('Error: cannot open camera')
        sys.exit(1)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('Camera - press q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Face detection using OpenCV Haar cascades')
    parser.add_argument('--source', '-s', default='camera', help="'camera' or path to an image file")
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index for webcam')
    parser.add_argument('--output', '-o', help='Output path for annotated image (image mode only)')
    parser.add_argument('--no-display', action='store_true', help='Do not display GUI windows')
    args = parser.parse_args()

    if args.source.lower() == 'camera':
        process_camera(args.camera)
    else:
        input_path = Path(args.source)
        process_image(input_path, output_path=args.output, show=not args.no_display)


if __name__ == '__main__':
    main()
