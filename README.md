# Indoor Covid Tracking System - Graduation Project

This project is a system designed to control access to indoor spaces using **facial recognition** and **mask detection** to limit the spread of **COVID-19**. The system automatically grants or denies access by detecting whether an individual is wearing a mask.

## Overview

The **Indoor Covid Tracking System** is built on a Raspberry Pi 4 and leverages **OpenCV**, **TensorFlow**, and **Keras** to perform facial recognition and mask detection. The system utilizes a camera to capture faces, and based on whether the person is wearing a mask, it either unlocks the door or denies access. 

### Key Features:
- **Facial Recognition**: Uses pre-trained models to recognize authorized individuals.
- **Mask Detection**: Ensures individuals are wearing a mask before granting entry.
- **Automated Door Lock Control**: Unlocks or locks a door using an electric solenoid lock.
- **LCD Display**: Provides feedback to the user (e.g., "Access Granted" or "Please wear a mask").

## Components Used:
- **Raspberry Pi 4 (4 GB RAM)**: The central processor for running the system.
- **Raspberry Pi Camera v1.3**: Captures video for facial recognition.
- **LCD 16x2 Display**: Displays messages for user interaction.
- **Relay Module**: Controls the electric solenoid lock.
- **Electric Solenoid Lock**: Used for controlling access to doors.
- **VNC Viewer**: Remotely controls the Raspberry Pi.

## Software and Libraries:
The system uses **Python** with the following libraries:
- **OpenCV**: For face detection and image processing.
- **Dlib**: For facial landmark detection.
- **Keras**: For deep learning-based mask detection.
- **TensorFlow**: For running the mask detection model.
- **Numpy**: For numerical operations related to image processing.
- **RPi.GPIO**: For controlling GPIO pins on the Raspberry Pi (used to control the relay).

## Setup Instructions:

1. **Clone the repository**:
   Download or clone this repository to your Raspberry Pi.

2. **Install system dependencies**:
   Run the following commands to install the necessary libraries and dependencies:

   ```bash
   sudo apt-get update
   sudo apt-get upgrade
   sudo apt install cmake build-essential pkg-config git
   sudo apt install libjpeg-dev libtiff-dev libjasper-dev libpng-dev libwebp-dev libopenexr-dev
   sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libdc1394-22-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
   sudo apt install libgtk-3-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
   sudo apt-get install libqt5gui5 libqt5webkit5 libqt5test5
   sudo apt install libatlas-base-dev liblapacke-dev gfortran
   sudo apt install libhdf5-dev libhdf5-103
   sudo apt install python3-dev python3-pip python3-numpy
   ```

3. **Install Python libraries**:

   ```bash
   pip install picamera[array]
   pip install opencv-python
   pip install opencv-contrib-python
   pip install numpy --upgrade
   pip install face-recognition
   pip install keras
   pip install tensorflow
   ```

4. **Install TensorFlow (for deep learning model)**:
   Follow the installation guide for TensorFlow on Raspberry Pi, as detailed in the previous section.

## Running the System:

To run the system, execute the main Python file (`code.py`):

```bash
python3 code.py
```

### System Workflow:
1. The system first performs **facial recognition** to check if the person is authorized.
2. If the person is recognized, the system proceeds to **mask detection**:
   - If the person is wearing a mask, the system grants access by unlocking the door.
   - If the person is not wearing a mask, the system denies access and displays a message to wear a mask.
3. The **LCD screen** provides feedback (e.g., "Access Granted" or "Please wear a mask").
4. The system will continuously run until manually stopped (by pressing 'ESC' in the OpenCV window).

### Code Breakdown:
1. **`faceRecognition()`**:
   - Loads known faces from the **Resources** folder and compares them with faces captured from the camera.
   - Displays the recognized name on the screen and updates a database with the person's entry time.
   
2. **`maskDetector()`**:
   - Loads a **face mask detection model** and uses it to determine whether a person is wearing a mask.
   - If the person is wearing a mask, the door is unlocked (via the **relay module**), and a message is displayed on the LCD.

### Example of Messages Displayed:
- **"Wear Your Mask"** – Displayed when the system is waiting for the user to wear a mask.
- **"Thanks, Now you can enter :)"** – Displayed after successful mask detection.
- **"Unwear Your Mask"** – Displayed before scanning for facial recognition.

## Troubleshooting:
- Ensure that the **camera** is properly connected and enabled.
- Double-check the **GPIO wiring** for the relay and solenoid lock.
- If the facial recognition or mask detection isn't working, verify that the training data for faces and the mask detection model are correctly loaded.

## License:
This project is open-source and licensed under the **MIT License**. You can freely fork, modify, and distribute it for personal or commercial use.

---

### Conclusion:
The **Indoor Covid Tracking System** helps control access to indoor spaces by verifying individuals' identities via **facial recognition** and ensuring they are wearing a mask, which is crucial for limiting the spread of **COVID-19**. The system uses cost-effective hardware and software to create a scalable, easy-to-implement solution for safer environments.
