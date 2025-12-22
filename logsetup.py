import logging


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures the root logger for console output.

    Args:
        level: The minimum logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-10.10s %(levelname)-4.4s %(message)s",
        datefmt="%M:%S",
    )
