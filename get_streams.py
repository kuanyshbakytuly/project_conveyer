import os
import cv2


def get_streams(
        cam_list:list,
        input_path:str,
        output_path:str
        ) -> tuple:
    """Функция, в которой реализован процесс создания объектов
        для последовательного считывания и записи кадров.

    Args:
        cam_list (list): Список камер(url-адресов к камерам)
        input_path (str): Временный параметр, эмуляция камер для работы
            с локальными файлами
        output_path (str): Путь для сохранения видео. Временный параметр.

    Returns:
        tuple: Список объектов, из которых считываем кадры(caps) и
            список объектов, которые записывают кадры(writers)
    """

    os.makedirs(output_path, exist_ok=True)

    caps = []
    writers = []

    # caps from each camera
    for cam in cam_list:
        cap = cv2.VideoCapture(f'{input_path}/{cam}')
        caps.append(cap)

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writers
        video_writer = cv2.VideoWriter(f'{output_path}/out_{cam}',
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       (w, h)
                                       )
        writers.append(video_writer)

    return caps, writers