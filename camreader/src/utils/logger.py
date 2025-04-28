import logging

def setup_logger():
    logger = logging.getLogger("RTSP_WebSocket")
    logger.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler("app.log")
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.INFO)

    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

logger = setup_logger()