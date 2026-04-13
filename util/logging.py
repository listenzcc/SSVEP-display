from loguru import logger

logger.add("logs/ssvep-display.log", rotation="1 MB")
