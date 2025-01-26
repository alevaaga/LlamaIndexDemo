from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.readers.file import UnstructuredReader
from pandas import DataFrame
from six import StringIO
from unstructured.documents.elements import Element

import crayon
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
import os

from unstructured.partition.utils.constants import OCR_AGENT_PADDLE

os.environ["OCR_AGENT"] = OCR_AGENT_PADDLE #"unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"


def load_documents(base_dir: Path) -> Dict[str, List[Document]]:
    Path(crayon.CACHE_ROOT).mkdir(parents=True, exist_ok=True)

    pkl_file = Path(crayon.CACHE_ROOT) / f"{str(Path(base_dir).name)}.pkl"
    if not pkl_file.is_file():
        import pickle

        company_docs = defaultdict(list)
        for sourcefile in base_dir.glob('**/*'):
            pages = read_pdf(sourcefile)
            for page, metadata, tables in pages:
                doc = Document(
                    text=page,
                    metadata=metadata
                )
                filename = doc.metadata.get("filename", "")
                year = int(filename.split("_")[-1].split(".")[0])
                company = filename.split("_")[0]
                num_tokens = Settings.tokenizer(doc.text)
                doc.metadata["company"] = company
                doc.metadata["year"] = year
                doc.metadata["num_tokens"] = len(num_tokens)
                docs = company_docs[year]
                docs.append(doc)

        with open(pkl_file, 'wb') as f:
            pickle.dump(company_docs, f)
    else:
        import pickle
        with open(pkl_file, 'rb') as inp:
            company_docs = pickle.load(inp)

    return company_docs


def read_pdf(sourcefile) -> List[Tuple[str, Dict[str, str], List[str]]]:
    # elements = partition_pdf(
    #     filename=sourcefile,
    #     infer_table_structure=True,
    #     extract_images_in_pdf=True,
    #     strategy="hi_res")

    elements = partition(
        filename=str(sourcefile),
        infer_table_structure=True,
        extract_images_in_pdf=True,
        strategy="hi_res")

    pages = []
    page_text = ""
    el = elements[0]
    current_page = el.metadata.page_number
    tables = []
    metadata = {
        "file_directory": el.metadata.file_directory,
        "filename": el.metadata.filename,
        "filetype": el.metadata.filetype,
        "languages": el.metadata.languages,
        "last_modified": el.metadata.last_modified,
        "page_number": current_page,
        # "tables": tables,
    }
    for el in elements:
        page_number = el.metadata.page_number
        if page_number != current_page:
            pages.append((page_text, metadata, tables))
            page_text = ""
            current_page = page_number
            tables = []
            metadata = {
                "file_directory": el.metadata.file_directory,
                "filename": el.metadata.filename,
                "filetype": el.metadata.filetype,
                "languages": el.metadata.languages,
                "last_modified": el.metadata.last_modified,
                "page_number": current_page,
                # "tables": tables,
            }

        if el.category == "Table":
            text, table_json = handle_table(el)
            tables.append(table_json)
            page_text += text
        elif el.category == "Title":
            page_text += "# " + el.text + "\n"
        elif el.category == "Header":
            page_text += "## " + el.text + "\n"
        elif el.category == "NarrativeText":
            page_text += el.text + "\n"
        elif el.category == "Image":
            page_text += el.text + "\n"
    else:
        pages.append((page_text, metadata, tables))

    return pages


def handle_table(el: Element) -> Tuple[str, str]:
    try:
        df = pd.read_html(StringIO(el.metadata.text_as_html), keep_default_na=False)[0]
        text = df.to_markdown() + "\n"
        return text, df.to_json()
    except Exception as e:
        return el.metadata.text_as_html, ""
