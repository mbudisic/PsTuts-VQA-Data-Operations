import glob
import json
from langchain_core.document_loaders import BaseLoader
from typing import List, Dict, Iterator, Tuple
from langchain_core.documents import Document

import aiofiles
import asyncio
from pathlib import Path
import requests
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)


def load_json_string(content: str, group: str):
    payload: List[Dict] = json.loads(content)
    [video.update({"group": group}) for video in payload]
    return payload


async def load_single_json(filepath):
    my_path = Path(filepath)

    async with aiofiles.open(my_path, mode="r", encoding="utf-8") as f:
        content = await f.read()
        payload = load_json_string(content, my_path.name)

    return payload


def load_VQA_file_from_url(url: str) -> Tuple[List[Document], str, List[Dict]]:
    """
    Loads a VQA dataset file from a URL and processes it into documents.

    Args:
        url (str): URL pointing to a JSON file containing VQA dataset

    Returns:
        tuple: A tuple containing:
            - list[Document]: Processed documents from the VideoTranscriptBulkLoader
            - str: Group name extracted from the URL filename
            - List[Dict]: The raw JSON payload loaded from the URL

    Note:
        This function needs to be updated as it currently has a type mismatch.
        The return type annotation indicates List[Document] but it returns a tuple.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    group = url.split("/")[-1].split(".")[0]
    json_payload = load_json_string(resp.content.decode("utf-8"), group)
    docs = VideoTranscriptBulkLoader(json_payload=json_payload).load()
    return docs, group, json_payload


async def load_json_files(path_pattern: List[str]):
    files = []
    for f in path_pattern:
        (files.extend(glob.glob(f, recursive=True)))

    tasks = [load_single_json(f) for f in files]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]  # flatten


class VideoTranscriptBulkLoader(BaseLoader):
    """Loads video transcripts as a bulk into documents"""

    def __init__(self, json_payload: List[Dict]):

        self.json_payload = json_payload

    def lazy_load(self) -> Iterator[Document]:
        """Lazy loader that returns an iterator"""

        for video in self.json_payload:
            metadata = dict(video)
            metadata.pop("transcripts", None)
            metadata.pop("qa", None)
            # Rename 'url' key to 'source' in metadata if it exists
            if "url" in metadata:
                metadata["source"] = metadata.pop("url")
            yield Document(
                page_content="\n".join(
                    t["sent"] for t in video["transcripts"]
                ),
                metadata=metadata,
            )


class VideoTranscriptChunkLoader(BaseLoader):
    """Loads video transcripts as individual chunks into documents"""

    def __init__(self, json_payload: List[Dict]):

        self.json_payload = json_payload

    def lazy_load(self) -> Iterator[Document]:
        """Lazy loader that returns an iterator"""

        for video in self.json_payload:
            metadata = dict(video)
            transcripts = metadata.pop("transcripts", None)
            metadata.pop("qa", None)
            # Rename 'url' key to 'source' in metadata if it exists
            if "url" in metadata:
                metadata["source"] = metadata.pop("url")
            for transcript in transcripts:
                yield Document(
                    page_content=transcript["sent"],
                    metadata=metadata
                    | {
                        "time_start": transcript["begin"],
                        "time_end": transcript["end"],
                    },
                )
