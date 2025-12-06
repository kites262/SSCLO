from loguru import logger


def setup_logger():
    LOG_FORMAT = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS}"
        "|"
        "<green>{level: <6}</green>"
        "|"
        "<cyan>{module: <20}</cyan>:<cyan>{line: <4}</cyan>"
        "> "
        "{message}"
    )
    logger.remove()
    logger.add(
        sink="train.log",
        format=LOG_FORMAT,
        enqueue=True,
        mode="w",
        level="DEBUG",
    )
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format=LOG_FORMAT,
        colorize=True,
        level="INFO",
    )
