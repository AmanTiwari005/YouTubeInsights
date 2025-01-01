import os
import cv2
import pytesseract
import torch
import yt_dlp
import whisper
import streamlit as st
from pytube import YouTube
from torchvision import transforms
from torchvision.models import detection

import yt_dlp
import os

import yt_dlp
import os

import yt_dlp
import os

# Function to download YouTube video using yt-dlp without merging
def download_video(youtube_url, output_dir="videos", cookies_path=None):
    os.makedirs(output_dir, exist_ok=True)

    # Options for yt-dlp to download the best available video and audio separately (no merging)
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Download the best video and audio separately
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),  # Template for saving the file
        'noplaylist': True,  # To avoid downloading entire playlist if URL is a playlist
    }

    # If cookies are provided, add them to yt-dlp options
    if cookies_path:
        ydl_opts['cookies'] = cookies_path

    # Using yt-dlp to download the video
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # Get the video path
    video_id = youtube_url.split("v=")[-1]
    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    return video_path


# Function to extract frames
def extract_frames(video_path, frame_rate=1, output_dir="frames"):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video_frames_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * frame_rate  # Extract one frame per second

    frame_count = 0
    saved_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(video_frames_dir, f"frame_{saved_frames}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
        frame_count += 1

    cap.release()
    return video_frames_dir

# Function to transcribe audio
def extract_audio_transcription(video_path, model_name="base"):
    model = whisper.load_model(model_name)
    transcription = model.transcribe(video_path)
    return transcription["text"]

# Function to detect objects in frames
def detect_objects_in_frames(frame_dir, model=None, confidence_threshold=0.5):
    if model is None:
        model = detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()

    transform = transforms.Compose([transforms.ToTensor()])
    results = {}

    for frame_name in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, frame_name)
        image = cv2.imread(frame_path)
        image_tensor = transform(image)

        with torch.no_grad():
            predictions = model([image_tensor])

        detected_objects = []
        for i, box in enumerate(predictions[0]['boxes']):
            score = predictions[0]['scores'][i].item()
            if score >= confidence_threshold:
                label = predictions[0]['labels'][i].item()
                detected_objects.append((label, score))

        results[frame_name] = detected_objects

    return results

# Function to extract text from frames
def extract_text_from_frames(frame_dir):
    results = {}
    for frame_name in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, frame_name)
        image = cv2.imread(frame_path)
        text = pytesseract.image_to_string(image)
        results[frame_name] = text

    return results

def main():
    st.title("YouTube Video Content Analysis")
    st.write("Enter the URLs of YouTube videos to analyze their content.")

    # Input: YouTube URLs
    video_urls = st.text_area("Enter YouTube URLs (one per line)").splitlines()

    if st.button("Analyze Videos"):
        if not video_urls:
            st.error("Please provide at least one YouTube video URL.")
            return

        # Create output directories
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Specify the path to your cookies file (for age-restricted videos)
        cookies_path = 'path_to_your_cookies_file'  # Replace with actual path if necessary

        # Process each video
        for url in video_urls:
            st.write(f"Processing video: {url}")
            try:
                # Download video
                video_path = download_video(url, output_dir=os.path.join(output_dir, "videos"), cookies_path=cookies_path)
                st.write(f"Video downloaded: {video_path}")

                # Extract frames
                frames_dir = extract_frames(video_path, output_dir=os.path.join(output_dir, "frames"))
                st.write(f"Frames extracted to: {frames_dir}")

                # Transcribe audio
                transcription = extract_audio_transcription(video_path)
                st.write("Transcription:")
                st.text_area("Transcription Text", transcription, height=150)

                # Object detection
                st.write("Performing object detection...")
                object_detection_results = detect_objects_in_frames(frames_dir)
                st.json(object_detection_results)

                # Text extraction
                st.write("Extracting text from frames...")
                text_extraction_results = extract_text_from_frames(frames_dir)
                st.json(text_extraction_results)

            except Exception as e:
                st.error(f"Error processing video {url}: {e}")

if __name__ == "__main__":
    main()
