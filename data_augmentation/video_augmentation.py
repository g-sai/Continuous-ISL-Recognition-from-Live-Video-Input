import os
import cv2
import random
import time
from imgaug import augmenters as iaa
from concurrent.futures import ThreadPoolExecutor

def augment_and_save_frames(video_reader, output_path, video_clip_name, i, fps, w, h):
    
    """
    Augments video frames by applying random transformations and saves the augmented frames to a new video file.
    
    Args:
        video_reader (cv2.VideoCapture): OpenCV VideoCapture object to read the video.
        output_path (str): Path where the augmented video will be saved.
        video_clip_name (str): Original name of the video clip.
        i (int): Index for the augmented video clip.
        fps (int): Frames per second of the video.
        w (int): Width of the video frames.
        h (int): Height of the video frames.

    """

    temp = video_clip_name.replace(" ", "")
    temp = temp.split(".")
    editted_name = f"{temp[0]}_{i}.{temp[1]}"
    path_of_video_to_save = os.path.join(output_path, editted_name)
    # flip = i % 2 == 0
    rotation_angle = random.randint(-25, 25)
    brightness_change = random.uniform(0.9, 1.1)
    contrast_change = random.uniform(0.9, 1.2)
    blur_strength = random.uniform(0, 0.5)
    noise_value = random.uniform(5, 13)

    seq = iaa.Sequential([
        # iaa.Fliplr(flip),
        iaa.Affine(rotate=rotation_angle),
        iaa.AdditiveGaussianNoise(scale=noise_value),
        iaa.Multiply(brightness_change),
        iaa.LinearContrast(contrast_change),
        iaa.GaussianBlur(sigma=blur_strength)
    ])

    fourcc = 'mp4v'
    video_writer = cv2.VideoWriter(path_of_video_to_save, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    speed_change_factor = random.uniform(0.8, 1.0)

    try:
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if not ret:
                break
        
            image_aug = seq(image=frame)
            video_writer.write(image_aug)
           

            time.sleep(1.0 / (fps * speed_change_factor))

    except Exception as e:
        print(f"Error processing video {video_clip_name}: {e}")

    finally:
        cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()


def augment_videos(input_path, output_path, video_clip_name, i):

    """
    Handles the process of augmenting a single video clip by opening the video and calling the augmentation function.
    
    Args:
        input_path (str): Directory containing the input video.
        output_path (str): Directory to save the augmented video.
        video_clip_name (str): The name of the video clip to augment.
        i (int): Index for the augmented video clip.
    
    """
    
    try:
        video_path = os.path.join(input_path, video_clip_name)
        video_reader = cv2.VideoCapture(video_path)
        fps = int(video_reader.get(cv2.CAP_PROP_FPS))
        w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Processing {video_clip_name} (FPS: {fps})")
        start = time.time()
        augment_and_save_frames(video_reader, output_path, video_clip_name, i, fps, w, h)
        end = time.time()
        print(f"Time taken for {video_clip_name}: {end-start:.2f} seconds")
    except Exception as e:
        print(f"Error augmenting {video_clip_name}: {e}")


def process_directory(input_dir, output_dir, no_of_augmentations):

    """
    Processes all video files in a directory, performing augmentation for each file.

    Args:
        input_dir (str): Path to the input directory containing the video files.
        output_dir (str): Path to the output directory to save augmented videos.
        no_of_augmentations (int): Number of augmented videos to generate for each input video.

    """

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):  
                relative_path = os.path.relpath(root, input_dir)
                input_path = root
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(output_path, exist_ok=True)
               
                print(f"Processing video: {file}")
                with ThreadPoolExecutor() as executor:
                    executor.map(lambda i: augment_videos(input_path, output_path, file, i), range(no_of_augmentations))



if __name__ == '__main__':
    main_folder_path = "/path/to/isl_dataset_dir"
    output_folder_path = "/path/to/augmented_isl_dataset_dir"
    no_of_clips_to_augment_per_video = 10

    os.makedirs(output_folder_path, exist_ok=True)

    start_time = time.time()
    process_directory(main_folder_path, output_folder_path, no_of_clips_to_augment_per_video)
    end_time = time.time()
   
    print(f"Total processing time: {end_time - start_time:.2f} seconds")




