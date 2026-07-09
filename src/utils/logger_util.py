# src/utils/logger_util.py
import sys
import logging
from termcolor import colored
from functools import lru_cache
from pathlib import Path
from datetime import datetime
import inspect

class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, self.RESET)
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"

@lru_cache()
def get_logger(name="default", output_dir="/Volumes/T7Shield/PROJECTS/AniVirusInteractome/logs", log_type="Run", level="INFO"):
    """
    获取一个支持彩色输出和日志文件写入的 logger。
    :param name: logger 的名字（可多个模块共享）
    :param output_dir: 日志文件保存路径
    :param log_type: 文件名中附加的标识，比如 "Train"、"Eval"
    :param level: logging level，如 logging.INFO、logging.DEBUG
    :return: logging.Logger 实例
    """
    level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(level, logging.INFO)  # 默认INFO
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 避免重复输出
    caller_frame = inspect.stack()[1]
    caller_filename = Path(caller_frame.filename).stem

    if not logger.handlers:
        log_fmt = '[%(asctime)s | %(name)s | %(levelname)s] (%(filename)s:%(lineno)d): %(message)s'
        date_fmt = '%Y-%m-%d %H:%M:%S'

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColorFormatter(fmt=log_fmt, datefmt=date_fmt))
        logger.addHandler(console_handler)

        # File handler
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_filename = f"{caller_filename}_{log_type}_{timestamp}.log"
        log_file = output_dir / log_filename

        file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt=log_fmt, datefmt=date_fmt))
        logger.addHandler(file_handler)

    return logger
