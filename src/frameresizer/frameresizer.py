import cv2

def resize_frame(frame, scale=None, width=None, height=None):
    """
    Resize a frame based on the specified scale or width/height while maintaining the aspect ratio.

    Parameters:
    - frame: Input frame to be resized.
    - scale: Resizing scale factor (float).
    - width: Width of the resized frame (integer).
    - height: Height of the resized frame (integer).

    Returns:
    - Resized frame.
    """
    if scale is not None:
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
    elif width is not None:
        height = int((width / frame.shape[1]) * frame.shape[0])
    elif height is not None:
        width = int((height / frame.shape[0]) * frame.shape[1])

    # Resize the frame
    resized_frame = cv2.resize(frame, (width, height)) # type: ignore

    return resized_frame