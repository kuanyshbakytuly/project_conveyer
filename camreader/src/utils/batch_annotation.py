import cv2
import asyncio
import numpy as np
import concurrent.futures
from transliterate import translit
from ultralytics.utils.plotting import colors


executor = concurrent.futures.ThreadPoolExecutor()

async def frame_annotation(
    frame: np.array, boxes: list, confs: list, names: list
) -> np.array:
    color_mapping = {
        'phone': colors(10, bgr=True),
        'no_gloves': colors(1, bgr=True),
        'gloves': colors(2, bgr=True),
        'mask': colors(3, bgr=True),
        'no_mask': colors(4, bgr=True),
        'improperly': colors(5, bgr=True),
        'visor_back': colors(6, bgr=True),
        'visor_forward': colors(7, bgr=True),
        'head': colors(8, bgr=True),
        'person': colors(15, bgr=True),
        'Unknown': (0, 0, 255)
    }

    if len(boxes) == 0:
        return frame

    loop = asyncio.get_event_loop()

    def annotate_frame():
        for i in range(len(boxes)):
            if names[i] not in color_mapping.keys():
                color = (255, 0, 0)
            else:
                color = color_mapping[names[i]]
            name = translit(names[i], 'ru', reversed=True)
            cv2.rectangle(
                img=frame,
                pt1=(int(boxes[i][0]), int(boxes[i][1])),
                pt2=(int(boxes[i][2]), int(boxes[i][3])),
                color=color,
                thickness=3
            )
            cv2.putText(
                img=frame,
                text=f'{round(float(confs[i]), 2)} {name}',
                org=(int(boxes[i][0]), int(boxes[i][1])),
                fontFace=cv2.FONT_ITALIC,
                thickness=2,
                fontScale=1,
                lineType=cv2.LINE_4,
                color=color
            )
        return frame

    annotated_frame = await loop.run_in_executor(executor, annotate_frame)
    return annotated_frame

async def batch_annotation(
    frame: np.array, all_models_annot: list, annot_models: dict
) -> np.array:
    models = {
        'Masks': 1,
        'Visors': 2,
        'Phones': 3,
        'Jewels': 4,
        'Smoke_face_eat': 5,
    }

    if len(annot_models) == 0:
        return frame
    
    annot = [0] + [models[key] for key in annot_models]

    for i in range(len(all_models_annot)): # I - model index
        if all_models_annot[i] and i in annot:
            for j in range(len(all_models_annot[i])): # len(all_models_annot[i]) is always 1 - because no batch inference
                boxes, confs, names = all_models_annot[i][j]
                frame = await frame_annotation(frame, boxes, confs, names)

    return frame