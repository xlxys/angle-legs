# Body Tracking Leg Angle Extractor

This project extracts 3D coordinates and angles of the left and right legs from a video using MediaPipe Pose and saves the results to a CSV file.

## Requirements

- Python 3.7+
- [OpenCV](https://pypi.org/project/opencv-python/)
- [MediaPipe](https://pypi.org/project/mediapipe/)
- [Pandas](https://pypi.org/project/pandas/)
- [tqdm](https://pypi.org/project/tqdm/)
- Numpy

Install dependencies with:

```sh
pip install opencv-python mediapipe pandas tqdm numpy
```


## Usage
Run the script with:

- --video <path_to_video>: Path to the input video file (required).
- --display: (Optional) Show pose detection and angles in real time.
Example:

- Press q to quit the display window.

## Output
The script saves the extracted data to leg_angles.csv in the current directory.
Files

- datagen.py: Main script for processing videos.
leg_angles.csv: Output CSV file with frame-by-frame leg joint coordinates and angles.
- videos/: Directory for input video files.

## Notes
- Make sure your video files are accessible and the path is correct.
- The script uses MediaPipe's world coordinates for landmarks.