from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.readers.file import UnstructuredReader
import crayon


def load_documents(base_dir: Path) -> List[Document]:
    Path(crayon.CACHE_ROOT).mkdir(parents=True, exist_ok=True)

    pkl_file = Path(crayon.CACHE_ROOT) / f"{str(Path(base_dir).name)}.pkl"
    if not pkl_file.is_file():
        import pickle

        docs = SimpleDirectoryReader(
            input_dir=str(base_dir),
            file_extractor={
                ".html": UnstructuredReader(),
            }
        ).load_data(
            num_workers=2,
            show_progress=True
        )
        for doc in docs:
            filename = doc.metadata.get("filename", doc.metadata.get("file_path", ""))
            year = int(filename.split("_")[-1].split(".")[0])
            num_tokens = Settings.tokenizer(doc.text)
            doc.metadata["year"] = year
            doc.metadata["num_tokens"] = len(num_tokens)

        with open(pkl_file, 'wb') as f:
            pickle.dump(docs, f)
    else:
        import pickle
        with open(pkl_file, 'rb') as inp:
            docs = pickle.load(inp)

    return docs
