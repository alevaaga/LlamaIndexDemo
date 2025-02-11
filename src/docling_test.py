import os
from warnings import filterwarnings

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core import Document
from spacy.matcher.dependencymatcher import defaultdict

filterwarnings(action="ignore", category=UserWarning, module="pydantic")
filterwarnings(action="ignore", category=FutureWarning, module="easyocr")
# https://github.com/huggingface/transformers/issues/5486:
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SOURCE = "/home/alex/PycharmProjects/LlamaIndexDemo/data/Knowledgebase/Finance/Crayon/Crayon_annual-report_2017.pdf"


def main():
    accelerator_options = AcceleratorOptions(
        num_threads=8, device=AcceleratorDevice.CUDA
    )
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    # Enable the profiling to measure the time spent
    settings.debug.profile_pipeline_timings = True

    conversion_result = converter.convert(SOURCE)
    doc = conversion_result.document

    # List with total time per document
    doc_conversion_secs = conversion_result.timings["pipeline_total"].times

    tables = defaultdict(list)
    for t in doc.tables:
        tables[t.prov[0].page_no].append(t)

    pages = doc.pages

    ldcos = []
    for page in pages.values():
        if page.image is not None:
            print("Got image")
        table = tables[page.page_no]
        text = doc.export_to_markdown(page_no=page.page_no)
        m = {
            "mimetype": doc.origin.mimetype,
            "page_number": page.page_no,
            "filename": doc.origin.filename,
        }
        if doc.origin.uri:
            m["uri"] = doc.origin.uri
        if table:
            m["table"] = table

        ldcos.append(Document(text=text, metadata=m))

    print(f"Conversion secs: {doc_conversion_secs}")

    # reader = DoclingReader()
    # node_parser = MarkdownNodeParser()
    # docs = reader.load_data(SOURCE)
    #
    # print(len(docs))
    print("")


if __name__ == '__main__':
    main()