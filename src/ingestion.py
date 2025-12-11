"""
ingestion.py

Provides extract_text_from_pdf which accepts either:
- raw PDF bytes, or
- filesystem path to a PDF.

Returns a list of dicts: [{"page_num": int, "text": str}, ...].
"""
from typing import List, Dict, Union
import fitz
from PIL import Image
import io
import pytesseract

def open_pdf(path_or_bytes: Union[str, bytes]):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return fitz.open(stream=path_or_bytes, filetype="pdf")
    return fitz.open(path_or_bytes)

def extract_text_from_pdf(path_or_bytes: Union[str, bytes],
                          ocr_if_needed: bool = True,
                          dpi: int=200) -> List[Dict]:
    #Extract text from each page of a PDF. Accepts bytes or filepath.
    doc = open_pdf(path_or_bytes)
    pages = []
    for i, page in enumerate(doc, start=1):
        try:
            text = page.get_text().strip()
        except Exception:
            text = ""
        if not text and ocr_if_needed:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes()))
            try:
                text = pytesseract.image_to_string(img)
            except Exception:
                text = ""
        pages.append({"page_num": i, "text": text})
    doc.close()
    return pages
