# Third-party notices

This source tree contains SmartDoc application code, browser captures and fictional test fixtures. It does not vendor Python wheels, external JS bundles, fonts, OCR engines, trained weights or datasets. The studio icon paths and existing favicon are preserved from the project source; no standalone third-party asset with unknown provenance is included.

## Browser dependencies

The legacy recruitment templates load these from their respective CDNs. They are referenced rather than copied into the repository.

| Component | License and attribution |
| --- | --- |
| Chart.js 4.4.7 | [MIT, Chart.js contributors](https://github.com/chartjs/Chart.js/blob/v4.4.7/LICENSE.md) |
| Font Awesome Free 6.5.2 | [Fort Awesome: CC BY 4.0 for icons, SIL OFL 1.1 for fonts, MIT for other files](https://github.com/FortAwesome/Font-Awesome/blob/6.5.2/LICENSE.txt) |
| DM Sans | [SIL OFL 1.1, DM Sans Project Authors](https://github.com/google/fonts/blob/main/ofl/dmsans/OFL.txt) |
| Manrope | [SIL OFL 1.1, Manrope Project Authors](https://github.com/google/fonts/blob/main/ofl/manrope/OFL.txt) |

## Installed dependencies and separately acquired models

- [Tesseract](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE) and [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast/blob/main/LICENSE): Apache 2.0. The engine, trained data and original installer are not redistributed here.
- [TrOCR base](https://huggingface.co/microsoft/trocr-base-handwritten) and [TrOCR small](https://huggingface.co/microsoft/trocr-small-handwritten): The Base card identifies MIT; the Small card does not declare a checkpoint license. Small is not redistributed. No TrOCR weights or IAM data are shipped. The optional recruitment embedding model is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2); the model page identifies Apache 2.0. No embedding weights are shipped.
- [pypdfium2 5.14.0](https://pypi.org/project/pypdfium2/5.14.0/) provides PDF text extraction and page rendering. Its Python bindings are offered under Apache-2.0 or BSD-3-Clause. The installed wheel contains the [BSD-style PDFium license](https://pdfium.googlesource.com/pdfium/+/refs/heads/main/LICENSE) and notices for its bundled native dependencies. Preserve the wheel's notices when redistributing it.
- [ReportLab 5.1.0](https://pypi.org/project/reportlab/5.1.0/) is a BSD-licensed test dependency for generating synthetic PDFs. Its default dependencies are Pillow and charset-normalizer. Neither it nor the PDFium wheel is bundled here.
- Python packages are installed independently. Key upstream notices: [Django](https://github.com/django/django/blob/main/LICENSE), [python-docx](https://github.com/python-openxml/python-docx/blob/master/LICENSE), [openpyxl](https://pypi.org/project/openpyxl/), [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE), [python-dotenv](https://github.com/theskumar/python-dotenv/blob/main/LICENSE), [pytesseract](https://github.com/madmaze/pytesseract/blob/master/LICENSE), [Google Gen AI SDK](https://github.com/googleapis/python-genai/blob/main/LICENSE), [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt), [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE), [SentencePiece](https://github.com/google/sentencepiece/blob/master/LICENSE), [Sentence Transformers](https://github.com/huggingface/sentence-transformers/blob/main/LICENSE) and [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE). See installed distributions for version-specific and transitive notices.
