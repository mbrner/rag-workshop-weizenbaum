from enum import Enum
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode, DoclingDocument
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus
from tqdm.auto import tqdm


def convert_all_docling(folder: Path, output_folder: Path | None = None, as_docling_doc: bool = True):
    folder = Path(folder)
    output_folder = Path(output_folder) if output_folder else None
    assert folder.is_dir()

    pdfs = [*folder.glob("*.pdf")]
    doc_return = []
    pdfs_to_process = []
    if output_folder is None:
        pdfs_to_process = pdfs
    else:
        for pdf in pdfs:
            doc_file = output_folder / f"{pdf.stem}.json"
            if doc_file.exists():
                doc_return.append(DoclingDocument.load_from_json(doc_file))
            else:
                pdfs_to_process.append(pdf)
    if not pdfs_to_process:
        return doc_return
    pipeline_options = PdfPipelineOptions(
        enable_remote_services=False
    )
    pipeline_options.do_picture_description = False

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    result = doc_converter.convert_all(pdfs_to_process)
    if output_folder is None:
         return doc_return + [doc.document if doc.status == ConversionStatus.SUCCESS else None for doc in result]
    for i, doc in enumerate(result):
        pdf = pdfs_to_process[i]
        if doc.status == ConversionStatus.SUCCESS:
            if as_docling_doc:
                doc.document.save_as_json(
                    output_folder / f"{pdf.stem}.json",
                    image_mode=ImageRefMode.EMBEDDED,
                )
            doc.document.save_as_markdown(
                output_folder / f"{pdf.stem}.md",
                image_mode=ImageRefMode.PLACEHOLDER,
            )
            doc_return.append(doc.document)
        else:
            doc_return.append(None)
    return doc_return
            