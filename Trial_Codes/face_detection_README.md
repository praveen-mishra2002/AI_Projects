# Face Detection (OpenCV)

Quickface detection script using OpenCV Haar cascades.

Installation

```bash
python -m pip install -r requirements.txt
```

Usage

Run webcam live detection:

```bash
python face_detection.py --source camera
```

Detect faces in an image and save annotated output:

```bash
python face_detection.py --source path/to/image.jpg --output out.jpg
```

Press `q` to quit the webcam window.
