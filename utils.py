
import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Union, Optional

from paths import DATA_DIR, PUBLICATION_FPATH, ENV_FPATH, CONFIG_FILE_PATH

def load_publication(publication_external_id="yzN0OCQT7hUS"):
    """Loads the publication markdown file.

    Returns:
        Content of the publication as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    publication_fpath = Path(os.path.join(DATA_DIR, f"{publication_external_id}.json"))

    # Check if file exists
    if not publication_fpath.exists():
        raise FileNotFoundError(f"Publication file not found: {publication_fpath}")

    # Read and return the file content
    try:
        with open(publication_fpath, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise IOError(f"Error reading publication file: {e}") from e


def load_all_publications(publication_dir: str = DATA_DIR) -> list[str]:
    """Loads all the publication markdown files in the given directory.

    Returns:
        List of publication contents.
    """
    publications = []
    for pub_id in os.listdir(publication_dir):
        if pub_id.endswith(".json"):
            publications.append(load_publication(pub_id.replace(".json", "")))
    return publications


def load_yaml_config(file_path: Union[str, Path]) -> dict:
    """Loads a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there's an error parsing YAML.
        IOError: If there's an error reading the file.
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    # Read and parse the YAML file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e

def load_config(config_path: str = CONFIG_FILE_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
