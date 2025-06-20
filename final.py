import os
import subprocess
import streamlit as st
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings
import cv2
from mediapipe.python.solutions import face_detection
from fer import FER
from transformers import pipeline

# CSS for background image
page_bg = """
<style>
body {
    background-image: url("./static/home.png"); /* Add your image path */
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


# Configure ImageMagick path (Update this path if needed)
change_settings({"IMAGEMAGICK_BINARY": r"C:/Program Files/ImageMagick-7.1.1-Q16-HDRI/magick.exe"})

# Load Whisper model
model = whisper.load_model("medium")

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Initialize FER for emotion detection
emotion_detector = FER(mtcnn=True)

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

# Function to transcribe audio and generate VTT with sentiments
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

def process_faces_and_emotions(input_video):
    try:
        # Initialize MediaPipe Face Detection
        face_detector = face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        # OpenCV video capture
        cap = cv2.VideoCapture(input_video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file = f"{os.path.splitext(input_video)[0]}_processed.mp4"
        out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use MediaPipe for face detection
            results = face_detector.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    # Extract bounding box coordinates
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, box_w, box_h = (
                        int(bbox.xmin * w),
                        int(bbox.ymin * h),
                        int(bbox.width * w),
                        int(bbox.height * h),
                    )

                    # Ensure bounding box dimensions are within frame boundaries
                    if x >= 0 and y >= 0 and (x + box_w) <= w and (y + box_h) <= h:
                        # Extract the face for emotion detection using FER
                        face_roi = frame[y:y + box_h, x:x + box_w]

                        # Perform emotion detection only if face_roi is non-empty
                        if face_roi.size > 0:
                            analysis = emotion_detector.detect_emotions(face_roi)
                            if analysis:
                                emotions = analysis[0]["emotions"]
                                dominant_emotion = max(emotions, key=emotions.get)

                                # Draw bounding box and emotion label
                                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                                cv2.putText(
                                    frame, f"{dominant_emotion}",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (0, 255, 0),
                                    2
                                )

            # Write the processed frame to the output file
            if out is None:
                height, width, _ = frame.shape
                fps = cap.get(cv2.CAP_PROP_FPS)
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            out.write(frame)

        cap.release()
        if out is not None:
            out.release()

        return output_file
    except Exception as e:
        st.error(f"Error processing faces and emotions: {e}")
        return None

def add_subtitles_faces_and_emotions(input_video, segments):
    try:
        # Load the video
        original_clip = VideoFileClip(video_path)  # from uploaded video (with audio)
        emotion_clip = VideoFileClip(emotion_video)  # face-annotated video (no audio)

        # Use original audio
        emotion_clip = emotion_clip.set_audio(original_clip.audio)

        fps = emotion_clip.fps or 24  # Ensure fallback to 24 FPS
        duration = emotion_clip.duration
        output_file = f"{os.path.splitext(input_video)[0]}_final.mp4"  # Assigned early

        subtitle_clips = []

        for segment in segments:
            start_time = min(segment["start"], duration - 0.1)
            end_time = min(segment["end"], duration)
            text = segment["text"]
            sentiment_label = segment.get("sentiment_label", "Neutral")
            senti_orientation = f"Sentiment: {sentiment_label}"

            # Subtitle at the bottom center
            subtitle = TextClip(
                text,
                fontsize=35,
                color="white",
                bg_color="black",
                size=(emotion_clip.w, None),
                method="caption",
                font="Arial"
            ).set_position(("center", "bottom")).set_start(start_time).set_end(end_time)

            # Sentiment at top-left corner
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

        # Combine video and overlayed clips
        final_video = CompositeVideoClip([emotion_clip] + subtitle_clips, size=emotion_clip.size)
        final_video = final_video.set_audio(emotion_clip.audio)  # Keep original audio
        final_video = final_video.set_duration(duration)

        # Write final video
        final_video.write_videofile(
            output_file,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            fps=fps,
            preset="ultrafast"  # Optional for speed
        )

        return output_file
    except Exception as e:
        st.error(f"Error generating final video: {e}")
        return None


# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select an option:", ["Home", "Emosentia App"])

if option == "Home":
    st.header("Welcome to the Emosentia App!")
    st.markdown("""
    - **Purpose**: Analyze videos for subtitles, sentiments, faces, and emotions.
    - **Features**:
        1. Automatic transcription and sentiment detection.
        2. Real-time emotion detection in faces.
        3. Combines everything into a single annotated video.
    """)

elif option == "Emosentia App":
    st.title("Emosentia App: Analyze Your Videos")
    st.info("Upload a video file to analyze subtitles, sentiments, faces, and emotions.")

    # Keep your existing Emosentia App logic here.
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])



    if uploaded_video is not None:
        # Save the uploaded video
        video_path = f"uploaded_{uploaded_video.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success(f"Video successfully uploaded! File path: {video_path}")

        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                # Step 1: Generate VTT with subtitles and sentiments
                vtt_path, segments = generate_vtt_with_sentiments(video_path)

                if segments:
                    st.success(f"Subtitles and sentiments generated: {vtt_path}")

                    # Step 2: Process faces and emotions
                    emotion_video = process_faces_and_emotions(video_path)
                    if emotion_video:
                        st.success(f"Emotion-detected video generated: {emotion_video}")
                        st.video(emotion_video)

                    # Step 3: Add subtitles, sentiments, faces, and emotions
                    final_video = add_subtitles_faces_and_emotions(emotion_video or video_path, segments)
                    if final_video:
                        st.success(f"Final video generated: {final_video}")
                        st.video(final_video)
                        st.download_button(
                            label="Download Final Video",
                            data=open(final_video, "rb").read(),
                            file_name=os.path.basename(final_video),
                            mime="video/mp4"
                        )
