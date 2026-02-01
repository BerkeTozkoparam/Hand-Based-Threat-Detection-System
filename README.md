# Hand-Based Threat Detection

This project is a real-time **hand gesture and motion based threat detection system**
built with **MediaPipe** and **OpenCV**.

It analyzes hand gestures and movement speed to estimate a simple threat level.
When the threat level exceeds a threshold, the system automatically starts
video recording.

> This is **not** a fight detection system.  
> It is an experimental, rule-based **threat pre-alert mechanism**.

## Features
- Real-time hand detection (MediaPipe Hands)
- Hand gesture recognition (fist, open hand, pointing, etc.)
- Motion speed tracking
- Simple threat scoring
- Automatic video recording on high threat
- Live visual indicators

## How to Run
```bash
pip install -r requirements.txt
python security.py
```
Press q to exit.
Recorded videos are saved under security_records/.
Tech Stack
Python
OpenCV
MediaPipe
NumPy
