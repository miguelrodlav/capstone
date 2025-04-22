# Capstone Project: Remote Assistance with Real-Time Point Tracking

Capstone Project for IIA Master's degree that implements a real-time point tracking system for remote assistance via video calls.

## Project Overview

This system provides a remote assistance platform where users can mark specific points on a video stream that are automatically tracked in real-time. The main components are:

- **Next.js Client**: A web interface for video calls that allows users to mark points on the video stream for tracking. The client sends both the coordinates of the selected point and video frames to the server.

- **Python Server**: Handles real-time point tracking using Meta's CoTracker3 model (online version) and manages WebRTC signaling for video communication.

## Key Features

- Real-time video communication using WebRTC
- Interactive point selection on video stream
- Real-time point tracking using state-of-the-art computer vision models
- Efficient processing with CoTracker3's online mode (reduced memory usage)
- Responsive web interface

## Technical Stack

- **Frontend**: Next.js, WebRTC
- **Backend**: Python
- **Computer Vision**: Meta's CoTracker3 (online version)
- **Video Processing**: OpenCV

## Proof of Concept

The `poc` directory contains a Python script that demonstrates the tracking functionality using OpenCV for camera capture, allowing for testing of the tracking model independently from the main application.

## Getting Started

[Installation and setup instructions will be added here]
