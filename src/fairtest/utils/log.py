"""
FairTest logs
"""
import logging


def set_params(filename='fairtest.log', level=logging.INFO):
    """
    Set Logging Parameters

    Parameters
    ----------
    filename :
        the name of the log file
    level :
        the logging level
    """
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=filename, mode='w+')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(level)
