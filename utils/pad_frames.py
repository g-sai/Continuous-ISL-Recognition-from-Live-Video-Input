import cv2

def pad_video(input_video_path, output_video_path, target_frame_count):
    
    """
    Pad a video with the last frame until the target frame count is reached.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path where the output padded video will be saved.
        target_frame_count (int): The target frame count for the output video.

    """

    cap = cv2.VideoCapture(input_video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_frame_count <= original_frame_count:
        print(f"Target frame count ({target_frame_count}) is less than or equal to original frame count ({original_frame_count}). No padding needed.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    last_frame = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_count += 1
        last_frame = frame  

    while frame_count < target_frame_count:
        out.write(last_frame)
        frame_count += 1

    cap.release()
    out.release()

    print(f"Video padding complete. Output video saved to {output_video_path} with {frame_count} frames.")


input_video_path = '/path/to/video.mp4'  
output_video_path = '/path/to/output_video.mp4'  
target_frame_count = 100  # Set the target number of frames

pad_video(input_video_path, output_video_path, target_frame_count)
