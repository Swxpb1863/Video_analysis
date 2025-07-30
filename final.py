import os
import subprocess
import streamlit as st
import whisper
import cv2
from mediapipe.python.solutions import face_detection
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

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def detect_emotion_opencv_only(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (48, 48))
        
        # 🧠 Replace this logic with your own trained model
        emotion = np.random.choice(EMOTIONS)  # Dummy prediction
        
        results.append({
            'box': (x, y, w, h),
            'emotion': emotion
        })

    return results
# Load Whisper model
model = whisper.load_model("medium")

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Initialize FER for emotion detection
emotion_detector = detect_emotion_opencv_only(frame)

for result in emotions:
    x, y, w, h = result['box']
    emotion = result['emotion']
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


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

def draw_text_on_frame(frame, text, position, font_scale=1, color=(255,255,255), thickness=2, background_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position

    # Draw background box
    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y + 10), background_color, -1)

    # Draw text over the box
    cv2.putText(frame, text, (x + 5, y), font, font_scale, color, thickness, cv2.LINE_AA)

def add_subtitles_faces_and_emotions(input_video, segments):
    try:
        cap = cv2.VideoCapture(input_video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_file = f"{os.path.splitext(input_video)[0]}_final_annotated.mp4"
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        segment_index = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps

            # Check if current_time falls in the current segment
            while (segment_index < len(segments) and 
                   current_time > segments[segment_index]['end']):
                segment_index += 1

            if segment_index < len(segments):
                seg = segments[segment_index]
                if seg['start'] <= current_time <= seg['end']:
                    # Draw subtitle and sentiment
                    draw_text_on_frame(frame, seg['text'], position=(50, height - 50))
                    draw_text_on_frame(frame, f"Sentiment: {seg.get('sentiment_label', 'Neutral')}", position=(10, 50), font_scale=0.8, color=(0, 255, 255))

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        return output_file
    except Exception as e:
        st.error(f"Error generating final video without ImageMagick: {e}")
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
