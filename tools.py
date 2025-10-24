import os
import shutil
import zipfile
import requests
from typing import List
from paths import DATA_DIR
from langchain_core.tools import tool
import torch
import chromadb
from paths import VECTOR_DB_DIR
from langchain_huggingface import HuggingFaceEmbeddings



@tool
def download_and_extract_repo(repo_url: str) -> str:
    """Download a Git repository and extract it to a local directory.

    This tool downloads a Git repository as a ZIP file from GitHub or similar
    platforms and extracts it to a './data/repo' directory. It handles both 'main'
    and 'master' branch repositories automatically. If the repo directory
    already exists, it will be removed and replaced with the new download.

    Args:
        repo_url: The complete URL of the Git repository (e.g., https://github.com/user/repo)

    Returns:
        The path to the extracted repository directory if successful, or False if failed
    """
    output_dir = os.path.join(DATA_DIR, "repo")
    try:
        if os.path.exists(output_dir):
            print(f"Repository already exists in {output_dir}, removing it")
            shutil.rmtree(output_dir)

        # Create target directory
        os.makedirs(output_dir, exist_ok=True)

        # Convert repo URL to zip download URL
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]
        if repo_url.endswith("/"):
            repo_url = repo_url[:-1]

        download_url = f"{repo_url}/archive/refs/heads/main.zip"

        print(f"Downloading repository from {download_url}")

        retires = 3
        i = 0
        while i < retires:
            response = requests.get(download_url, stream=True)
            if response.status_code == 404:
                download_url = f"{repo_url}/archive/refs/heads/master.zip"
                response = requests.get(download_url, stream=True)

            if response.status_code != 200:
                print(f"Failed to download repository: {response.status_code}")
                i += 1
                continue

            response.raise_for_status()
            break

        temp_dir = os.path.join(output_dir, "_temp_extract")
        os.makedirs(temp_dir, exist_ok=True)

        temp_zip = os.path.join(temp_dir, "repo.zip")
        with open(temp_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find the nested directory (it's usually named 'repo-name-main')
        nested_dirs = [
            d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))
        ]
        if nested_dirs:
            nested_dir = os.path.join(temp_dir, nested_dirs[0])

            for item in os.listdir(nested_dir):
                source = os.path.join(nested_dir, item)
                destination = os.path.join(output_dir, item)
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)

        shutil.rmtree(temp_dir)

        return output_dir

    except requests.exceptions.RequestException as e:
        print(f"Failed to download repository: {str(e)}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        return False

    except zipfile.BadZipFile as e:
        print(f"Invalid zip file: {str(e)}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        return False

    except OSError as e:
        print(f"OS error occurred: {str(e)}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        return False

    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        return False


@tool
def env_content(dir_path: str) -> str:
    """Read and return the content of a .env file from a specified directory.

    This tool searches through the given directory path and its subdirectories
    to find a .env file and returns its complete content. Useful for examining
    environment variables and configuration settings.

    Args:
        dir_path: The directory path to search for .env file (must be a local path, not URL)

    Returns:
        The complete content of the .env file as a string, or None if not found
    """
    for dir, _, files in os.walk(dir_path):
        for file in files:
            if file == ".env":
                with open(os.path.join(dir, file), "r") as f:
                    return f.read()
    return None


@tool
def retrieve_relevant_documents(    query: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> list[str]: 
    """
    Query the ChromaDB database with a string query.

    Args:
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)
        threshold (float): Threshold for the cosine similarity score (default: 0.3)

    Returns:
        dict: Query results containing ids, documents, distances, and metadata
    """
    # logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }
    # Embed the query using the same model used for documents
    # logging.info("Embedding query...")
    # Ashok 1
    def embed_documents(documents: list[str]) -> list[list[float]]:
        """
        Embed documents using a model.
        """
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
        )

        embeddings = model.embed_documents(documents)
        return embeddings
    # Ashok 1
    # logging.info("Querying collection...")
    # Query the collection
    query_embedding = embed_documents([query])[0]  # Get the first (and only) embedding
   #ashok 2
    def get_db_collection(
        persist_directory: str = './vector_db',
        collection_name: str = "publications",
    ) -> chromadb.Collection:
        """
        Get a ChromaDB client instance.

        Args:
            persist_directory (str): The directory where ChromaDB persists data
            collection_name (str): The name of the collection to get

        Returns:
            chromadb.PersistentClient: The ChromaDB client instance
        """
        return chromadb.PersistentClient(path=persist_directory).get_collection(
            name=collection_name
        )
    
    collection = get_db_collection(collection_name="publications")
    # ashok2
    results = collection.query(
         query_embeddings=[query_embedding],
         n_results=n_results,
         include=["documents", "distances"],
     )
# ashok
    # logging.info("Filtering results...")
    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    return relevant_results["documents"]


# @tool
# def embed_documents(documents: list[str]) -> list[list[float]]:
#     """
#     Embed documents using a model.
#     """
#     device = (
#         "cuda"
#         if torch.cuda.is_available()
#         else "mps" if torch.backends.mps.is_available() else "cpu"
#     )
#     model = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": device},
#     )

#     embeddings = model.embed_documents(documents)
#     return embeddings

# @tool
# def get_db_collection(
#     persist_directory: str = VECTOR_DB_DIR,
#     collection_name: str = "publications",
# ) -> chromadb.Collection:
#     """
#     Get a ChromaDB client instance.

#     Args:
#         persist_directory (str): The directory where ChromaDB persists data
#         collection_name (str): The name of the collection to get

#     Returns:
#         chromadb.PersistentClient: The ChromaDB client instance
#     """
#     return chromadb.PersistentClient(path=persist_directory).get_collection(
#         name=collection_name
#     )

@tool
def hello_tool(name: str) -> str:
    """A simple test tool to confirm that the tool call works in the graph.

    This tool takes a name as input and returns a greeting message.
    Useful for verifying that the Agentic AI Graph can call custom tools correctly.

    Args:
        name: The name of the person or entity to greet.

    Returns:
        A simple greeting message confirming the tool was executed.
    """
    return f"Hello, {name}! The tool has been called successfully 🎯"

def get_all_tools() -> List:
    """Return a list of all available tools."""
    return [
        env_content,
        download_and_extract_repo,
        retrieve_relevant_documents,
        hello_tool

    ]