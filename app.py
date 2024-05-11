import os
import cv2
import numpy as np
import streamlit as st
from inference_sdk import InferenceHTTPClient
import requests

def analyze_video_frame(frame_path, roboflow_api_key, model_name):
    """
    Analyze a video frame to detect and track the ball, including ground impact detection.
    
    Args:
    - frame_path (str): Path to the video frame image file.
    - roboflow_api_key (str): RoboFlow API key.
    - model_name (str): Name of the deployed model on RoboFlow.
    
    Returns:
    - dict: Information about ball tracking and ground impact detection.
    """
    # URL for RoboFlow inference API
    url = f"https://detect.roboflow.com/{model_name}?api_key={roboflow_api_key}"
    
    # Open the image file
    with open(frame_path, 'rb') as image_file:
        # Making the request to RoboFlow API
        response = requests.post(url, files={'file': image_file})
        result = response.json()
    
    # Assume the model can detect the ball's position
    ball_positions = []
    
    for detection in result['predictions']:
        if detection['class'] == 'ball':
            ball_positions.append((detection['x'], detection['y']))
    
    # Analyze ball positions for ground impact detection
    ground_impact_detected = False
    if len(ball_positions) > 1:
        # Compare current and previous ball positions
        current_position = ball_positions[-1]
        previous_position = ball_positions[-2]
        if current_position[1] > previous_position[1] + 10:  # Adjust threshold as needed
            ground_impact_detected = True
    
    return {
        'ball_positions': ball_positions,
        'ground_impact_detected': ground_impact_detected
    }

def create_colored_heatmaps(frames_folder, output_image_path, roboflow_api_key, model_name):
    # Load the base image to overlay the heatmaps
    base_image = cv2.imread(output_image_path)
    
    # Check the image size and create heatmaps accordingly
    heatmap1 = np.zeros((base_image.shape[0], base_image.shape[1]), dtype=np.float32)
    heatmap2 = np.zeros((base_image.shape[0], base_image.shape[1]), dtype=np.float32)

    # Initialize the client
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=roboflow_api_key
    )

    # Iterate through each frame in the folder and accumulate heatmaps
    for filename in os.listdir(frames_folder):
        if filename.endswith(".jpg"):
            frame_path = os.path.join(frames_folder, filename)
            if st.session_state['track_ball']:
                analysis_result = analyze_video_frame(frame_path, roboflow_api_key, model_name="training_ball_squash")
                ball_positions = analysis_result['ball_positions']

                # Process ball positions for ground impact detection
                if analysis_result['ground_impact_detected']:
                    # If the ball hits the ground, mark the impact point on the heatmap
                    for position in ball_positions:
                        x, y = position
                        heatmap1[y, x] += 1  # Example update, adjust as needed
            else:
                result = CLIENT.infer(frame_path, model_id="training_player_detection")
                image = cv2.imread(frame_path)

                # Scale the frame to match the base image size if needed
                if image.shape != base_image.shape:
                    image = cv2.resize(image, (base_image.shape[1], base_image.shape[0]))

                for prediction in result['predictions']:
                    xmin = int(prediction['x'] * base_image.shape[1] / image.shape[1])
                    ymin = int(prediction['y'] * base_image.shape[0] / image.shape[0])
                    xmax = xmin + int(prediction['width'] * base_image.shape[1] / image.shape[1])
                    ymax = ymin + int(prediction['height'] * base_image.shape[0] / image.shape[0])

                    if prediction['class'] == 'squash-players1':
                        heatmap2[ymin:ymax, xmin:xmax] += prediction['confidence']
                    elif prediction['class'] == 'squash-players2':
                        heatmap2[ymin:ymax, xmin:xmax] += prediction['confidence']

    # Normalize and apply colormap to each heatmap
    heatmap1 = cv2.normalize(heatmap1, None, 0, 255, cv2.NORM_MINMAX)
    heatmap2 = cv2.normalize(heatmap2, None, 0, 255, cv2.NORM_MINMAX)
    colored_heatmap1 = cv2.applyColorMap(heatmap1.astype(np.uint8), cv2.COLORMAP_JET)
    colored_heatmap2 = cv2.applyColorMap(heatmap2.astype(np.uint8), cv2.COLORMAP_JET)

    # Blend the heatmaps with the original image
    overlayed_image1 = cv2.addWeighted(base_image, 0.7, colored_heatmap1, 0.3, 0)
    overlayed_image2 = cv2.addWeighted(base_image, 0.7, colored_heatmap2, 0.3, 0)

    # Display the final images using Streamlit
    st.image(overlayed_image1, caption='Ball Impact Heatmap Overlayed Image', use_column_width=True)
    st.image(overlayed_image2, caption='Player Detection Heatmap Overlayed Image', use_column_width=True)

# Streamlit UI
st.title("Squash Player Detection")

# Allow the user to upload a video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file is not None:
    st.session_state['track_ball'] = st.checkbox("Track Ball")
    if st.session_state['track_ball']:
        st.session_state['model_name'] = "training_ball_squash"
    else:
        st.session_state['model_name'] = "training_player_detection"
    # Save the uploaded video file
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Extract frames from the uploaded video
    cap = cv2.VideoCapture("uploaded_video.mp4")
    frame_folder = "uploaded_frames"
    os.makedirs(frame_folder, exist_ok=True)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frame_folder, f"frame{frame_count}.jpg"), frame)
        frame_count += 1
    cap.release()

    # Process the frames and create heatmaps
    create_colored_heatmaps(frame_folder, "path/to/your/base/image.png", "your_roboflow_api_key", st.session_state['model_name'])
