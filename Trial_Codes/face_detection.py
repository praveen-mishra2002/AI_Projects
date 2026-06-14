#!/usr/bin/env python3
"""Simple face detection using OpenCV Haar cascades.

Supports webcam live detection and single-image annotation.
"""
import argparse
import sys
from pathlib import Path
import os
import urllib.request

import cv2
import numpy as np


def _nms(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2.argsort()
    while len(idxs) > 0:
        last = idxs[-1]
        i = last
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:-1]]
        idxs = np.delete(idxs, np.concatenate(([len(idxs)-1], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")


def detect_faces(image, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)):
    """Detect faces using multiple Haar cascades and non-max suppression.

    Returns an array of (x, y, w, h) boxes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_names = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt.xml',
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_alt_tree.xml',
    ]
    boxes = []
    for name in cascade_names:
        cascade_path = cv2.data.haarcascades + name
        detector = cv2.CascadeClassifier(cascade_path)
        dets = detector.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        for (x, y, w, h) in dets:
            boxes.append((x, y, w, h))
    if len(boxes) == 0:
        return []
    import numpy as np
    boxes_np = np.array(boxes)
    # apply non-max suppression to reduce duplicates
    final = _nms(boxes_np, overlapThresh=0.3)
    return final


def _ensure_dnn_model(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    proto = os.path.join(model_dir, 'deploy.prototxt')
    model = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    proto_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
    model_url = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
    if not os.path.exists(proto):
        print('Downloading DNN prototxt...')
        urllib.request.urlretrieve(proto_url, proto)
    if not os.path.exists(model):
        print('Downloading DNN caffemodel (this may take a while)...')
        urllib.request.urlretrieve(model_url, model)
    return proto, model


def load_dnn(net_dir=None):
    if net_dir is None:
        net_dir = Path(__file__).parent / 'models'
    proto, model = _ensure_dnn_model(str(net_dir))
    net = cv2.dnn.readNetFromCaffe(proto, model)
    return net


def detect_faces_dnn(image, net, conf_threshold=0.5, nms_thresh=0.3):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            x = max(0, startX)
            y = max(0, startY)
            bw = min(w - 1, endX) - x
            bh = min(h - 1, endY) - y
            if bw > 0 and bh > 0:
                boxes.append((x, y, bw, bh))
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes)
    # apply non-max suppression to DNN boxes to remove overlaps
    final = _nms(boxes_np, overlapThresh=nms_thresh)
    return final



def draw_faces(image, faces, color=(0, 255, 0), thickness=2):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def process_image(input_path, output_path=None, show=True, backend='haar', net=None, conf=0.5, nms=0.3):
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: couldn't read image '{input_path}'")
        sys.exit(1)
    if backend == 'dnn':
        faces = detect_faces_dnn(image, net, conf_threshold=conf, nms_thresh=nms)
    else:
        faces = detect_faces(image)
    draw_faces(image, faces)
    if output_path:
        cv2.imwrite(str(output_path), image)
    if show:
        cv2.imshow('Face Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"Detected {len(faces)} face(s)")


def process_camera(camera_index=0, backend='haar', net=None, conf=0.5, nms=0.3):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print('Error: cannot open camera')
        sys.exit(1)
    # use the unified detect_faces implementation for consistency
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if backend == 'dnn':
                faces = detect_faces_dnn(frame, net, conf_threshold=conf, nms_thresh=nms)
            else:
                faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('Camera - press q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Face detection using OpenCV Haar cascades or DNN')
    parser.add_argument('--source', '-s', default='camera', help="'camera' or path to an image file")
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index for webcam')
    parser.add_argument('--output', '-o', help='Output path for annotated image (image mode only)')
    parser.add_argument('--no-display', action='store_true', help='Do not display GUI windows')
    parser.add_argument('--backend', '-b', choices=['haar', 'dnn'], default='dnn', help='Detection backend to use')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for DNN backend')
    parser.add_argument('--nms', type=float, default=0.3, help='NMS overlap threshold for DNN backend')
    parser.add_argument('--model-dir', help='Directory to store/download DNN model files')
    args = parser.parse_args()
    net = None
    if args.backend == 'dnn':
        net_dir = Path(args.model_dir) if args.model_dir else None
        net = load_dnn(net_dir)

    if args.source.lower() == 'camera':
        process_camera(args.camera, backend=args.backend, net=net, conf=args.conf, nms=args.nms)
    else:
        input_path = Path(args.source)
        process_image(input_path, output_path=args.output, show=not args.no_display, backend=args.backend, net=net, conf=args.conf, nms=args.nms)


if __name__ == '__main__':
    main()
