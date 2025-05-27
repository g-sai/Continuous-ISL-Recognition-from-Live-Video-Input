import cv2

def get_video_shape(video_path):

    """
    Retrieves the shape of a video, including the total number of frames, frame height, and frame width.

    Args:
        video_path (str): The path to the video file.

    Returns:
        tuple: A tuple containing (frame_count, frame_height, frame_width) if successful, otherwise None.
    """

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return frame_count, frame_height, frame_width

video_path = '/path/to/video'
video_shape = get_video_shape(video_path)

if video_shape:
    frame_count, frame_height, frame_width = video_shape
    print(f"Number of frames: {frame_count}")
    print(f"Frame height: {frame_height}, Frame width: {frame_width}")



