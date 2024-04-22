from pathlib import Path
from typing import Dict

import requests
from llama_index.core import SimpleDirectoryReader, Document

wiki_titles = [
    "Earnings per share",
    "Financial ratio",
    "Sharpe ratio",
    "Debt ratio",
    "Loan-to-value ratio",
    "P/B ratio",
]


def get_wikipedia_finance() -> Dict[str, Document]:
    data_path = Path("data/finance/wikipedia")

    for title in wiki_titles:
        output_filename = data_path / f"{title.replace('/', '_')}.txt"
        if not output_filename.exists():
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "extracts",
                    # 'exintro': True,
                    "explaintext": True,
                },
            ).json()
            page = next(iter(response["query"]["pages"].values()))
            wiki_text = page["extract"]

            Path.mkdir(data_path, parents=True, exist_ok=True)

            with open(output_filename, "w") as fp:
                fp.write(wiki_text)

    # Load all wiki documents
    docs_dict = {}
    for wiki_title in wiki_titles:
        doc = SimpleDirectoryReader(
            input_files=[f"{data_path}/{wiki_title.replace('/', '_')}.txt"]
        ).load_data()[0]

        # doc.metadata.update(wiki_metadatas[wiki_title])
        docs_dict[wiki_title] = doc
    return docs_dict
