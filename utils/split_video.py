import cv2
import os

def split_video(input_video_path, frame_segments, output_dir="/path/to/default_output_dir"):
    
    """
    Splists a video into multiple parts based on the provided frame segments.
    
    Args:
        input_video_path (str): Path to the input video file.
        frame_segments (list of tuple): List of tuples where each tuple contains (start_frame, end_frame) to define splits.
        output_dir (str): Directory to save the split video files..

    """

    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames in video: {total_frames}")
    print(f"Video FPS: {fps}, Width: {width}, Height: {height}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (start_frame, end_frame) in enumerate(frame_segments):
        end_frame = min(end_frame, total_frames)
        
        split_filename = f"{output_dir}/split_{i+1}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(split_filename, fourcc, fps, (width, height))
        
        print(f"Processing split {i+1}: frames {start_frame} to {end_frame}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_no in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()

    cap.release()



input_video_path = "/path/to/video.mp4"
frame_segments = [(0, 73), (73, 128), (128, 205)]  # Required list of tuples (start_frame, end_frame)
output_dir = "/path/to/output_dir"  

split_video(input_video_path, frame_segments, output_dir)
