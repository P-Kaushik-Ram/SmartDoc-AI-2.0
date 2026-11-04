"""Generate synthetic PDF test inputs independently of the production reader."""
from io import BytesIO

from reportlab.pdfgen.canvas import Canvas


def pdf_bytes(pages, *, page_size=(595, 842), encrypt=None):
    buffer = BytesIO()
    canvas = Canvas(buffer, pagesize=page_size, encrypt=encrypt)
    for text in pages:
        if text:
            canvas.drawString(40, page_size[1] - 72, text)
        canvas.showPage()
    canvas.save()
    return buffer.getvalue()
