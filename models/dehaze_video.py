import cv2
from models.dehaze_image import dehaze_image

def dehaze_video(input_path, output_path, omega=0.85, progress_callback=None):
    """
    Dehaze a video frame-by-frame.

    input_path  : path to the hazy input video
    output_path : where to save the dehazed video
    omega       : haze removal strength (passed to dehaze_image)
    progress_callback : optional function(fraction) for UI progress bars
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")

    # Read the video's properties so the output matches
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up the output writer ('mp4v' codec for .mp4 files)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # no more frames

        # Reuse our image dehazer on each frame
        dehazed = dehaze_image(frame, omega=omega)
        writer.write(dehazed)

        # Report progress (0.0 -> 1.0) if a callback was given
        frame_index += 1
        if progress_callback and total_frames > 0:
            progress_callback(frame_index / total_frames)

    # Always release resources
    cap.release()
    writer.release()

    return output_path
