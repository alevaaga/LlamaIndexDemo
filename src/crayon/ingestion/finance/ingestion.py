import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from six import StringIO
from tqdm import tqdm
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition
from unstructured.partition.utils.constants import OCR_AGENT_PADDLE

import crayon

os.environ["OCR_AGENT"] = OCR_AGENT_PADDLE #"unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"
MIN_SIZE = 300

def load_documents(base_dir: Path) -> Dict[str, List[BaseNode]]:
    Path(crayon.CACHE_ROOT).mkdir(parents=True, exist_ok=True)

    pkl_file = Path(crayon.CACHE_ROOT) / f"{str(Path(base_dir).name)}.pkl"
    if not pkl_file.is_file():
        import pickle

        company_docs = defaultdict(list)
        progress_bar = tqdm(base_dir.glob('**/*'), total=len(list(base_dir.glob('**/*'))))
        for sourcefile in progress_bar:
            progress_bar.set_description(f"Processing {sourcefile}")
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
                nodes = SentenceSplitter.from_defaults(chunk_size=1024, chunk_overlap=0).get_nodes_from_documents([doc])
                company_docs[year].extend(nodes)

        with open(pkl_file, 'wb') as f:
            pickle.dump(company_docs, f)
    else:
        import pickle
        with open(pkl_file, 'rb') as inp:
            company_docs = pickle.load(inp)

    for year in company_docs:
        company_docs[year] = list(filter(lambda doc: doc.metadata["num_tokens"] > MIN_SIZE, company_docs[year]))

    return company_docs


def read_pdf(sourcefile) -> List[Tuple[str, Dict[str, str], List[str]]]:
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
