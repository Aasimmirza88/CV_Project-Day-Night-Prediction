import cv2
import numpy as np

def classify_frame(frame):
    """
    Classify a single frame into Day, Evening, or Night based on brightness.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    avg_brightness = np.mean(gray)  # Calculate the average brightness

    if avg_brightness > 150:  # Brightness threshold for "Day"
        return "Day"
    elif 80 <= avg_brightness <= 150:  # Brightness range for "Evening"
        return "Evening"
    else:  # Low brightness indicates "Night"
        return "Night"

def VideoClassification(video_path, output_path):
    """
    Classify video frames into Day, Evening, or Night based on brightness
    and save the processed video with classifications displayed on frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_cnt = 0
    day_cnt, evening_cnt, night_cnt = 0, 0, 0

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_cnt += 1

        # Classify the frame
        classification = classify_frame(frame)
        if classification == "Day":
            day_cnt += 1
        elif classification == "Evening":
            evening_cnt += 1
        elif classification == "Night":
            night_cnt += 1

        # Overlay the classification on the frame
        cv2.putText(frame, classification, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Write the processed frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate and print classification percentages
    if frame_cnt > 0:
        print(f"Total Frames: {frame_cnt}")
        print(f"Day: {day_cnt / frame_cnt * 100:.2f}%")
        print(f"Evening: {evening_cnt / frame_cnt * 100:.2f}%")
        print(f"Night: {night_cnt / frame_cnt * 100:.2f}%")
    else:
        print("No frames were processed.")

# File paths
video_path = "E:/My_project/Day_Night/input/5763215-hd_1920_1080_30fps.mp4"
output_path = "E:/My_project/Day_Night/input/output/pred_video.mp4"

# Call the function
VideoClassification(video_path, output_path)
