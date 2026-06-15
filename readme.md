# OpenCV Hand Tracker - Air Writing Board

This is a Python-based **Air Writing Board** project built using **OpenCV** and **MediaPipe**.
It allows the user to write on the screen using one finger and erase the drawing using an open palm gesture.

## Features

* Write in air using index finger
* Erase using open palm gesture
* Real-time hand tracking using webcam
* Drawing canvas using OpenCV
* Save drawing as an image
* Clear canvas using keyboard shortcut
* Uses MediaPipe Hand Landmarker model

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy

## How It Works

The webcam detects your hand in real time.

* When only the **index finger** is up, the app enters writing mode.
* When the **full palm** is open, the app enters erasing mode.
* The drawing is displayed directly on the webcam screen.

## Controls

| Key | Action           |
| --- | ---------------- |
| `q` | Quit the app     |
| `c` | Clear the canvas |
| `s` | Save the drawing |

## Requirements

Use **Python 3.12** for best compatibility.

Install the required libraries:

```bash
pip install opencv-python mediapipe numpy
```

## Model File Required

This project uses the MediaPipe Hand Landmarker model.

Download the model file and place it in the same folder as `df.py`.

Model file name:

```text
hand_landmarker.task
```

You can download it using PowerShell:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/akshat2007-sys/open-cv-handtracker.git
```

Go inside the project folder:

```bash
cd open-cv-handtracker
```

Run the Python file:

```bash
python df.py
```

## Project Structure

```text
open-cv-handtracker/
│
├── df.py
├── hand_landmarker.task
└── README.md
```

## Notes

If the camera does not open, change this line in the code:

```python
self.cap = cv2.VideoCapture(0)
```

to:

```python
self.cap = cv2.VideoCapture(1)
```

Use good lighting for better hand detection.

## Developed By

**Akshat Srivastava**
