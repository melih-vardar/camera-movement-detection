import streamlit as st
import numpy as np
from PIL import Image
import movement_detector
import cv2
import os
import tempfile

st.title("Camera Movement Detection Demo by Melih Vardar")
st.write(
    "Upload a sequence of images (e.g., from a camera). The app will detect frames with significant camera movement."
)

tab1, tab2 = st.tabs(["Upload Images", "Upload Video"])

with tab1:
    st.subheader("Upload Multiple Images")
    uploaded_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        frames = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            frame = np.array(image)

            if frame.shape[-1] == 4:  # Checking if it is RGBA
                frame = frame[:, :, :3] # RGBA to RGB

            frames.append(frame)

        st.write(f"Loaded {len(frames)} frames.")
        
        # unless there is more than one frame, do not run the detector
        if len(frames) > 1:
            
            movement_indices = movement_detector.detect_significant_movement(frames)
            
            st.write("Significant movement detected at frames:", movement_indices)

            # show frames with detected movement
            if movement_indices:
                st.subheader("Frames with Movement")
                for idx in movement_indices:
                    st.image(frames[idx], caption=f"Movement at frame {idx}", use_column_width=True)
            else:
                st.info("No movement detected.")
        else:
            st.info("Please upload at least 2 images.")

with tab2:
    st.subheader("Upload Video")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video:
        # temporary file (for creating a path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name

        try:
            # extracting frames
            st.write("Processing video...")
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # processing frames at certain intervals
            frame_step = max(1, total_frames // 50)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # frame step control
                if frame_count % frame_step == 0:
                    # BGR --> RGB (opencv to streamlit format)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                frame_count += 1
            
            cap.release()

            if frames:
                st.write(f"Extracted {len(frames)} frames.")

                # process frames for movement detection
                if len(frames) > 1:
                    
                    movement_indices = movement_detector.detect_significant_movement(frames)

                    if movement_indices:
                        movement_percentage = len(movement_indices) / len(frames) * 100
                        st.write(f"Movement detected in {len(movement_indices)}/{len(frames)} frames ({movement_percentage:.1f}%)")

                        st.subheader("Frames with Movement")
                        # show first 3 frames
                        for i, idx in enumerate(movement_indices[:3]):
                            st.image(frames[idx], caption=f"Frame {idx} - Movement detected")
                
                    else:
                        st.info("No movement detected.")
                        
                else:
                    st.error("Not enough frames extracted for movement detection.")
        
            else:
                st.error("Failed to extract frames from video.")

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        ## temporary file will be deleted if exists
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

                
                


                
