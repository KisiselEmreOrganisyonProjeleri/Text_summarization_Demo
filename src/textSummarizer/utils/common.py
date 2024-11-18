import os  # Provides functions to interact with the operating system
from box.exceptions import BoxValueError  # Exception raised for issues in Box library operations
import yaml  # Library to parse YAML files
from src.textSummarizer import logger  # Logger setup for the project
from ensure import ensure_annotations  # Decorator to enforce type annotations
from box import ConfigBox  # ConfigBox class from Box for managing dictionary-like structures
from pathlib import Path  # Library to handle file system paths
from typing import Any  # For general type hints

@ensure_annotations  # Ensures that annotations are followed for this function
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns its content as a ConfigBox

    Args:
        path_to_yaml (Path): Path object pointing to the YAML file location

    Raises:
        ValueError: raised if the YAML file is empty
        e: raises any other exceptions encountered

    Returns:
        ConfigBox: Loaded content wrapped in ConfigBox for attribute-style access
    """
    try:
        # Opens the YAML file in read mode
        with open(path_to_yaml) as yaml_file:
            # Loads the content using yaml.safe_load, which prevents execution of arbitrary code
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")  # Logs successful load
            return ConfigBox(content)  # Returns the content wrapped as a ConfigBox
    except BoxValueError:
        # Raises ValueError if YAML file is empty or unreadable by ConfigBox
        raise ValueError("yaml file is empty")
    except Exception as e:
        # Catches any other exception and raises it
        raise e

@ensure_annotations  # Enforces annotations for create_directories function
def create_directories(path_to_directories: list, verbose=True):
    """Creates multiple directories from a list of paths

    Args:
        path_to_directories (list): List containing directory paths to create
        verbose (bool, optional): If True, logs each directory creation. Defaults to True.
    """
    for path in path_to_directories:  # Iterates over each directory path in the list
        os.makedirs(path, exist_ok=True)  # Creates directory if it does not exist
        if verbose:  # Logs creation if verbose is True
            logger.info(f"created directory at: {path}")

@ensure_annotations  # Enforces type annotations for get_size function
def get_size(path: Path) -> str:
    """Calculates and returns file size in KB

    Args:
        path (Path): Path object pointing to the file

    Returns:
        str: Formatted size in KB
    """
    # Calculates file size in KB by dividing bytes by 1024 and rounds it
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"  # Returns size as a string in KB