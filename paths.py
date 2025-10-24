import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

DATA_DIR = os.path.join(ROOT_DIR, "myBooks/output_unstruct")

CONFIG_DIR = os.path.join(ROOT_DIR, "LangGraph_Joke/config")

CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "config.yaml")
PROMPT_CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "prompt_config.yaml")
VECTOR_DB_DIR = os.path.join(ROOT_DIR, "myBooks/vector_db")
PUBLICATION_FPATH = os.path.join(DATA_DIR, "publication.md")
ENV_FPATH = os.path.join(ROOT_DIR, ".env")