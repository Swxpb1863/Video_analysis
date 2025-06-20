import os
import time
import streamlit as st
import whisper
import subprocess
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings
from transformers import pipeline

# Configure ImageMagick path (Update this path if needed)
change_settings({"IMAGEMAGICK_BINARY": r"C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe"})

# Load Whisper model
model = whisper.load_model("medium")

# Load Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Sentiment label mapping
LABEL_MAPPING = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

# Function to convert video to audio (mp3)
def video_to_audio(video_file, output_ext="mp3"):
    audio_file = f"{os.path.splitext(video_file)[0]}.{output_ext}"
    subprocess.call(
        ["ffmpeg", "-y", "-i", video_file, audio_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    return audio_file

# Function to transcribe and save subtitles with sentiments
def generate_vtt_with_sentiments(input_video):
    try:
        # Convert video to audio
        audio_file = video_to_audio(input_video)

        # Transcribe audio
        result = model.transcribe(audio_file)

        # Prepare .vtt content with sentiments
        vtt_path = f"{os.path.splitext(input_video)[0]}.vtt"
        segments = result["segments"]

        with open(vtt_path, "w") as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            for segment in segments:
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                text = segment["text"]

                # Analyze sentiment for the segment
                sentiment_result = sentiment_analyzer(text)[0]
                sentiment_label = LABEL_MAPPING.get(sentiment_result["label"], sentiment_result["label"])
                segment["sentiment_label"] = sentiment_label  # Add to segments for later use

                # Write caption and sentiment in .vtt
                vtt_file.write(f"{start} --> {end}\n")
                vtt_file.write(f"{text.strip()}\n")
                vtt_file.write(f"Sentiment: {sentiment_label}\n\n")

        return vtt_path, segments
    except Exception as e:
        st.error(f"Error generating VTT with sentiments: {e}")
        return None, None

# Helper function to format timestamp to VTT-compatible format
def format_timestamp(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02}:{mins:02}:{secs:02}.{millis:03}"

# Function to add subtitles and sentiments to a video
def add_subtitles_with_sentiments(input_video, segments):
    try:
        # Load the video
        video_clip = VideoFileClip(input_video)

        # Create subtitle and sentiment clips
        subtitle_clips = []
        for segment in segments:
            start_time, end_time = segment["start"], segment["end"]
            text = segment["text"]
            sentiment_label = segment["sentiment_label"]
            senti_orientation = f"Sentiment: {sentiment_label}"
            # Subtitle at the bottom center
            subtitle = TextClip(
                text,
                fontsize=35,
                color="white",
                bg_color="black",
                size=(video_clip.w, None),
                method="caption",
                font="Arial"
            ).set_position(("center", "bottom")).set_start(start_time).set_end(end_time)

            # Sentiment at the top left corner
            sentiment = TextClip(
                senti_orientation,
                fontsize=25,
                color="yellow",
                bg_color="black",
                size=(250, None),
                font="Arial"
            ).set_position((10, 10)).set_start(start_time).set_end(end_time)

            subtitle_clips.append(subtitle)
            subtitle_clips.append(sentiment)

        # Combine the video and clips
        final_video = CompositeVideoClip([video_clip] + subtitle_clips)

        # Save the final video
        output_file = f"{os.path.splitext(input_video)[0]}_subtitled.mp4"
        final_video.write_videofile(output_file, codec="libx264", audio_codec="aac")

        return output_file
    except Exception as e:
        st.error(f"Error generating subtitled video: {e}")
        return None

# Streamlit app UI
st.title("Add Subtitles and Sentiment Analysis to Your Video")
st.info("Upload a video file to generate subtitles and perform sentiment analysis.")

# File uploader
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video
    video_path = f"uploaded_{uploaded_video.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.success(f"Video successfully uploaded! File path: {video_path}")

    # Generate subtitles and subtitled video
    if st.button("Generate Subtitled Video and Sentiments"):
        with st.spinner("Processing your video..."):
            # Step 1: Generate .vtt with captions and sentiments
            vtt_path, segments = generate_vtt_with_sentiments(video_path)

            if segments:
                st.success(f"Subtitles with sentiments generated: {vtt_path}")

                # Step 2: Add subtitles and sentiments to video
                subtitled_video = add_subtitles_with_sentiments(video_path, segments)

                if subtitled_video:
                    st.success(f"Subtitled video generated: {subtitled_video}")
                    st.video(subtitled_video)
                    st.download_button(
                        label="Download Subtitled Video",
                        data=open(subtitled_video, "rb").read(),
                        file_name=os.path.basename(subtitled_video),
                        mime="video/mp4"
                    )
                else:
                    st.error("Failed to add subtitles to the video.")
