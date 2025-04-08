import numpy as np
import cv2


def get_frames(caps:cv2.VideoCapture) -> list:
    """Формирование списка кадров из списка объектов Videocapture

    Args:
        caps (cv2.VideoCapture): Считанное видео

    Returns:
        list: Список кадров
    """

    dummy_frame = np.zeros((640, 640, 3))
    frames = []

    # frames reading(from each camera)
    for i in range(len(caps)):
        success, frame = caps[i].read()
        if success:
            frames.append(frame)
        else:
            frames.append(dummy_frame)
            # logger.debug(f'For {cam_list[i]} - dummy_frame')

    return frames