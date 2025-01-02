# YouTube Video Content Analysis

This repository contains a comprehensive Python application for analyzing the content of YouTube videos. The application allows you to download videos, extract frames, transcribe audio, detect objects in frames, and extract text from frames using a combination of powerful libraries and models.

## Features

- **Video Downloading**: Downloads videos from YouTube using `yt-dlp`.
- **Frame Extraction**: Extracts frames from downloaded videos at a specified frame rate.
- **Audio Transcription**: Transcribes audio from videos using the `whisper` model.
- **Object Detection**: Detects objects in extracted frames using a pre-trained Faster R-CNN model.
- **Text Extraction**: Extracts text from frames using `pytesseract`.

## Requirements

To run this application, you need to have the following libraries installed:

- `os`
- `cv2` (OpenCV)
- `pytesseract`
- `torch`
- `yt_dlp`
- `pytube`
- `whisper`
- `streamlit`
- `torchvision`

You can install the required libraries using the following command:

```bash
pip install opencv-python pytesseract torch yt-dlp pytube whisper streamlit torchvision
