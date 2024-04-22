from loguru import logger

logger.add(
    "logs/{time:YYYY-MM-DD}.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG",
)


def debug(message: str):
    logger.debug(message)


def info(message: str):
    logger.info(message)


def warning(message: str):
    logger.warning(message)


def error(message: str):
    logger.error(message)


def critical(message: str):
    logger.critical(message)


def exception(message: str):
    logger.exception(message)
