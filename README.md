# Drowsy Driver Detection

A real-time fatigue detection application for classifying drowsy and non-drowsy drivers. Capstone project for Computer Vision class.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [License](#license)

## Introduction

The Fatigue Detection System is designed to monitor the user's alertness by analyzing their eye aspect ratio (EAR)  and mouth aspect ratio (MAR) through a webcam. It provides visual and audible alerts when signs of fatigue are detected, thereby reducing the risk of accidents caused by drowsiness.

## Features

- Real-time eye tracking to detect fatigue
- Visual indicators showing the user's alertness status
- Audible alarms to alert the user

## Requirements

This application requires the following:

- Python 3.6+
- OpenCV
- Mediapipe
- dlib
- PyQt5
- numpy

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Luchanaaaaa/Fatigue-Detection-System.git
cd Fatigue-Detection-System
```
Install the required dependencies:
```
pip install -r requirements.txt
```
## Usage
To run the application, execute the following command:
```python
python main.py
```
Make sure your webcam is enabled and properly set up before starting the application!

## How it Works
- The application uses a webcam to continuously monitor the user.
- Facial landmarks are detected using dlib and Mediapipe's pre-trained model.
- Eye aspect ratios and mouth aspect ratios are calculated to determine the user's level of alertness.
- The system classifies the user's state as active, fatigued, or asleep based on EAR and MAR values and provides corresponding alerts.
  
## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
