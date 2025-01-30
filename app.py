import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

# Initialize the summarization pipeline using Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to get YouTube transcript with timestamps
def get_transcript_with_timestamps(video_url):
    try:
        video_id = video_url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Function to summarize text using Hugging Face with timestamps
def summarize_text_with_huggingface(transcript):
    try:
        # Combine text for summarization
        text = " ".join([t["text"] for t in transcript])
        
        # Split the text into chunks if it's too long (Hugging Face models have token limits)
        max_chunk_length = 1024  # Adjust based on the model's token limit
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        full_summary = " ".join(summaries)
        
        # Add timestamps to the summary
        timestamped_summary = []
        for segment in transcript:
            if any(keyword.lower() in segment["text"].lower() for keyword in full_summary.split()):
                timestamp = segment["start"]
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                timestamp_str = f"{minutes:02d}:{seconds:02d}"
                timestamped_summary.append(f"[{timestamp_str}] {segment['text']}")
        
        return full_summary, timestamped_summary
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None, None

# Streamlit app
def main():
    st.title("YouTube Video Summarizer")
    st.markdown("""
        This app summarizes YouTube videos using AI. Simply paste the URL of the video below and click 'Summarize' to get a concise summary with timestamps.
    """)
    
    # Input for YouTube video URL
    video_url = st.text_input("Enter YouTube video URL:")
    
    if st.button("Summarize"):
        if video_url:
            with st.spinner("Fetching transcript and summarizing..."):
                transcript = get_transcript_with_timestamps(video_url)
                if transcript:
                    summary, timestamped_summary = summarize_text_with_huggingface(transcript)
                    if summary and timestamped_summary:
                        st.success("Summary generated successfully!")
                        
                        # Display the summary
                        st.subheader("Summary")
                        st.write(summary)
                        
                        # Display the timestamped summary
                        st.subheader("Timestamped Summary")
                        for line in timestamped_summary:
                            st.write(line)
        else:
            st.warning("Please enter a valid YouTube video URL.")

if __name__ == "__main__":
    main()
