"""Optional, local TrOCR service for English handwritten text lines."""
import os
import threading
from pathlib import Path

from .utils import DocumentProcessingError, MAX_IMAGE_PIXELS

MODEL_ID = os.environ.get('SMARTDOC_HANDWRITING_MODEL', 'microsoft/trocr-base-handwritten')
MODEL_PATH_ENV = 'SMARTDOC_HANDWRITING_MODEL_PATH'
_BACKEND = None
_BACKEND_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _configured_model_source():
    """Resolve the explicitly configured local checkpoint, or the cache id."""
    configured_path = os.environ.get(MODEL_PATH_ENV, '').strip()
    if configured_path:
        model_path = Path(configured_path).expanduser()
        if not model_path.is_dir():
            raise DocumentProcessingError(
                f'The configured TrOCR model directory does not exist: {model_path}. '
                'Download the checkpoint explicitly with '
                f'hf download {MODEL_ID} --local-dir '
                '"$env:SMARTDOC_HANDWRITING_MODEL_PATH", then retry.'
            )
        return str(model_path.resolve())
    return MODEL_ID


def _load_backend():
    """Initialize once per Django worker; never download weights implicitly."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is not None:
            return _BACKEND
        model_source = _configured_model_source()
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise DocumentProcessingError(
                'Handwriting recognition is unavailable. Install the optional PyTorch and Transformers dependencies.'
            ) from exc
        try:
            import sentencepiece  # noqa: F401 - required by the TrOCR tokenizer
        except ImportError as exc:
            raise DocumentProcessingError(
                'Handwriting recognition is unavailable because SentencePiece is missing. '
                'Install requirements-handwriting.txt, then retry.'
            ) from exc

        device_name = os.environ.get('SMARTDOC_HANDWRITING_DEVICE', 'cpu').lower()
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_name.startswith('cuda') and not torch.cuda.is_available():
            raise DocumentProcessingError(
                'Handwriting recognition was configured for CUDA, but CUDA is unavailable. Set SMARTDOC_HANDWRITING_DEVICE=cpu or install a compatible CUDA-enabled PyTorch build.'
            )
        if device_name not in {'cpu', 'cuda'} and not device_name.startswith('cuda:'):
            raise DocumentProcessingError('SMARTDOC_HANDWRITING_DEVICE must be cpu, cuda, cuda:N, or auto.')

        try:
            processor = TrOCRProcessor.from_pretrained(model_source, local_files_only=True)
            model = VisionEncoderDecoderModel.from_pretrained(model_source, local_files_only=True)
            model.to(device_name)
            model.eval()
        except Exception as exc:
            message = str(exc).lower()
            if 'local entry not found' in message or 'not found in the cached files' in message or 'no such file' in message or 'does not appear to have a file' in message:
                raise DocumentProcessingError(
                    f'The TrOCR checkpoint is incomplete or unavailable locally. Download it explicitly with '
                    f'hf download {MODEL_ID} --local-dir "<local-model-directory>", set '
                    f'{MODEL_PATH_ENV} to that directory, then retry. No model download was attempted.'
                ) from exc
            raise DocumentProcessingError('The local TrOCR model could not be initialized. Check the configured checkpoint files, PyTorch installation, and available memory.') from exc
        _BACKEND = (torch, processor, model, device_name)
        return _BACKEND


def _page_ink(gray):
    """A detection mask only: never use this threshold to erase letter pixels."""
    import numpy as np
    from PIL import ImageFilter
    pixels = np.asarray(gray, dtype=np.float32)
    radius = max(2, min(30, round(max(gray.size) * 0.0125)))
    background = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    difference = background - pixels
    # Photographed textured paper can put noise above a fixed threshold on
    # every row, merging an entire page into one crop. Estimate that noise
    # robustly; this mask affects geometry only, never recognition pixels.
    noise = float(np.median(np.abs(difference)))
    textured = noise > 4
    if textured:
        # Only the layout mask is denoised on visibly textured paper.
        detection = gray.filter(ImageFilter.MedianFilter(3))
        pixels = np.asarray(detection, dtype=np.float32)
        background = np.asarray(detection.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
        difference = background - pixels
        noise = float(np.median(np.abs(difference)))
    threshold = max(12, min(36, noise * 3))
    ink = difference > threshold
    if textured:
        from scipy import ndimage
        labels, _ = ndimage.label(ink)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= 16
        keep[0] = False
        ink = keep[labels]
    return ink


def _remove_edge_artifacts(ink):
    """Discard long connected camera/page edges, retaining ordinary edge text."""
    import numpy as np
    from scipy import ndimage
    labels, _ = ndimage.label(ink)
    height, width = ink.shape
    removed = np.zeros_like(ink)
    objects = ndimage.find_objects(labels)
    for label, bounds in enumerate(objects, 1):
        if bounds is None:
            continue
        ys, xs = bounds
        w, h = xs.stop - xs.start, ys.stop - ys.start
        at_edge = xs.start == 0 or ys.start == 0 or xs.stop == width or ys.stop == height
        thin_edge = (w > width * .5 and h < height * .08 and
                     (ys.start < height * .06 or ys.stop > height * .94)) or (
                     h > height * .5 and w < width * .08 and
                     (xs.start < width * .06 or xs.stop > width * .94))
        corner = ((xs.start == 0 or xs.stop == width) and
                  (ys.start == 0 or ys.stop == height) and
                  w < width * .1 and h < height * .1 and np.count_nonzero(labels[bounds] == label) > 100)
        if corner or (thin_edge and (at_edge or w > width * .8 or h > height * .8)):
            # The threshold sees a dark frame's rim, not its solid interior.
            # Clear its narrow bounding strip too, or whitening the rim creates
            # a fresh high-contrast edge on the next detection pass.
            removed[bounds] = True
    # Once a frame is established, also exclude its disconnected fragments in
    # the outer 1% margin (JPEG edges often fade and split into components).
    if removed.any():
        for bounds in objects:
            if bounds is None:
                continue
            ys, xs = bounds
            if (xs.stop <= width * .01 or xs.start >= width * .99 or
                    ys.stop <= height * .01 or ys.start >= height * .99):
                removed[bounds] = True
            elif (xs.stop == width or xs.start == 0) and xs.stop - xs.start < width * .03:
                removed[bounds] = True
    return ink & ~removed, removed


def _orient_page(gray, ink, edges):
    """EXIF is handled first; infer quarter turns only for clearly vertical lines.

    Projection deskew uses a small mask, ignores borders, and requires a clear
    improvement. It cannot reliably decide whether handwriting is upside down.
    """
    import numpy as np
    from PIL import Image
    yy, xx = np.nonzero(ink)
    if not len(xx):
        return gray, ink, edges
    if np.ptp(yy) > max(1, np.ptp(xx)) * 2.5:
        gray = gray.rotate(-90, expand=True, fillcolor=255)
        ink = np.rot90(ink, -1)
        edges = np.rot90(edges, -1)
    mask = Image.fromarray(ink.astype('uint8') * 255)
    mask.thumbnail((700, 700))

    def score(angle):
        values = np.asarray(mask.rotate(angle, expand=True), dtype=np.float32) / 255
        projection = values.sum(axis=1)
        return float((projection ** 2).sum())

    original = score(0)
    angles = np.arange(-7, 7.01, .5)
    best = max(angles, key=score)
    if abs(best) >= .5 and score(best) > original * 1.12:
        gray = gray.rotate(float(best), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
        ink = np.asarray(Image.fromarray(ink).rotate(float(best), expand=True), dtype=bool)
        edges = np.asarray(Image.fromarray(edges).rotate(float(best), expand=True), dtype=bool)
    return gray, ink, edges


def preprocess_handwriting_image(image, with_layout=False):
    """Normalize a page and prepare individual text-line crops for TrOCR."""
    try:
        import numpy as np
        from PIL import Image, ImageFilter, ImageOps
        from scipy import ndimage
    except ImportError as exc:
        raise DocumentProcessingError('Handwriting preprocessing requires Pillow, NumPy and SciPy. Install requirements-handwriting.txt.') from exc

    if not isinstance(image, Image.Image):
        raise DocumentProcessingError('Handwriting recognition requires a valid image.')
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise DocumentProcessingError('Image dimensions exceed the 40-megapixel processing limit.')
    if image.width < 8 or image.height < 8:
        raise DocumentProcessingError('Image is too small to contain a readable handwritten line.')
    image = ImageOps.exif_transpose(image)
    # Composite transparent inputs onto paper instead of turning transparency black.
    if image.mode in {'RGBA', 'LA'} or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        paper = Image.new('RGBA', image.size, 'white')
        image = Image.alpha_composite(paper, rgba).convert('RGB')
    gray = ImageOps.grayscale(image)
    initial_ink, initial_edges = _remove_edge_artifacts(_page_ink(gray))
    # Transform the detection/exclusion masks alongside the source. Thresholding
    # again after whitening a gray paper edge would create an artificial seam.
    gray, ink, edge_mask = _orient_page(gray, initial_ink, initial_edges)
    image = gray
    pixels = np.asarray(gray, dtype=np.uint8)
    # Estimate uneven page illumination locally before thresholding. A global
    # threshold treated the photographed paper/shadows as ink and expanded
    # crops far beyond the handwritten marks.
    blur_radius = max(2, min(30, round(max(image.size) * 0.0125)))
    background = gray.filter(ImageFilter.GaussianBlur(blur_radius))
    # Tiny isolated flecks are excluded from detection only. The source grayscale
    # remains intact for recognition, including dots and faint connected strokes.
    labels, _ = ndimage.label(ink)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= 3
    keep[0] = False
    ink = keep[labels]

    def longest_run(values):
        padded = np.concatenate(([False], values, [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        return int((ends - starts).max()) if len(starts) else 0

    def ranges(flags):
        positions = np.flatnonzero(flags)
        if not len(positions):
            return []
        groups = []
        start = previous = int(positions[0])
        for position in positions[1:]:
            position = int(position)
            if position > previous + 1:
                groups.append((start, previous + 1))
                start = position
            previous = position
        groups.append((start, previous + 1))
        return groups

    def merge_ranges(groups, max_gap):
        merged = []
        for start, end in groups:
            if merged and start - merged[-1][1] <= max_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    def is_rule_only_band(region):
        """Reject a band dominated by one long slanted ruling stroke."""
        if region.shape[0] < max(16, int(image.height * 0.015)):
            # Resampling a rotated paper edge can leave a one-pixel band.
            # Keep ordinary short, small text (including detached dots).
            columns = np.flatnonzero(region.any(axis=0))
            span = int(columns[-1] - columns[0] + 1) if len(columns) else 0
            return span > max(200, region.shape[0] * 60)
        if (
            region.shape[1] <= region.shape[0] * 3
            or region.size > 2_000_000
        ):
            return False
        y, x = np.nonzero(region)
        if len(x) < 100:
            return False
        best = (0, 0.0, 0)
        for slope in np.linspace(-0.25, 0.25, 41):
            rho = np.rint(y - slope * x).astype(np.int32)
            counts = np.bincount(rho - rho.min())
            peak = int(np.convolve(counts, np.ones(7, dtype=np.int32), mode='same').max())
            if peak > best[0]:
                best = (peak, float(slope), int(np.argmax(np.convolve(
                    counts, np.ones(7, dtype=np.int32), mode='same'
                )) + rho.min()))
        if best[0] < len(x) * 0.72 or best[0] < max(80, region.shape[1] * 0.2):
            return False
        on_line = np.abs(y - best[1] * x - best[2]) <= 3
        return len(x) - int(on_line.sum()) < max(32, int(region.shape[1] * 0.12))

    # Long, nearly straight strokes are table borders/underlines, not text.
    # Remove them from the row projection and retain vertical dividers so rows
    # in ruled tables can be split into their separate cells.
    vertical_flags = np.fromiter(
        (longest_run(ink[:, x]) > max(40, int(image.height * 0.03)) for x in range(image.width)),
        dtype=bool, count=image.width,
    )
    horizontal_flags = np.fromiter(
        (longest_run(ink[y, :]) > max(50, int(image.width * 0.06)) for y in range(image.height)),
        dtype=bool, count=image.height,
    )
    # A photographed/ragged table rule can fade for a few columns. Join its
    # nearby flagged runs so it becomes one separator rather than tiny crops.
    vertical_candidates = merge_ranges(ranges(vertical_flags), max(6, round(image.width * 0.005)))
    horizontal_rules = ranges(horizontal_flags)
    vertical_rule_presence = []
    rule_mask = np.zeros_like(ink)
    rule_rows = np.zeros(image.height, dtype=bool)
    for start, end in vertical_candidates:
        x0, x1 = max(0, start - 2), min(image.width, end + 2)
        region = ink[:, x0:x1]
        row_support = region.sum(axis=1) >= max(2, int((x1 - x0) * 0.08))
        continuous_rows = np.zeros(image.height, dtype=bool)
        for top, bottom in ranges(row_support):
            if bottom - top > max(80, int(image.height * 0.12)):
                continuous_rows[top:bottom] = True
        if continuous_rows.any():
            vertical_rule_presence.append((start, end, continuous_rows))
            row_padding = max(2, round(image.height * 0.015))
            column_padding = max(6, round(image.width * 0.015))
            rule_top = max(0, int(np.flatnonzero(continuous_rows)[0]) - row_padding)
            rule_bottom = min(image.height, int(np.flatnonzero(continuous_rows)[-1]) + row_padding + 1)
            rule_mask[rule_top:rule_bottom,
                      max(0, start - column_padding):min(image.width, end + column_padding)] = True
    for start, end in horizontal_rules:
        rule_rows[max(0, start - 2):min(image.height, end + 2)] = True
    ink[rule_mask] = False
    ink[rule_rows, :] = False

    row_ink = ink.sum(axis=1)
    active_rows = row_ink >= max(3, int(image.width * 0.0015))
    bands = []
    start = previous = None
    # A slightly wider row gap keeps detached dots/descenders with their line.
    gap_limit = max(5, round(image.height * 0.006))
    for row, active in enumerate(active_rows):
        if active:
            if start is None:
                start = row
            elif row - previous > gap_limit:
                bands.append((start, previous + 1))
                start = row
            previous = row
    if start is not None:
        bands.append((start, previous + 1))

    # Preserve grayscale stroke detail; optionally flatten lighting smoothly.
    # No median/erosion filter is applied to handwriting itself.
    illumination = np.asarray(background, dtype=np.float32)
    if os.environ.get('SMARTDOC_HANDWRITING_NORMALIZE', 'false').lower() == 'false':
        cleaned = pixels.copy()
    else:
        cleaned = np.minimum(255, pixels.astype(np.float32) * 255 / np.maximum(illumination, 1)).astype(np.uint8)
    cleaned[edge_mask | rule_mask] = 255
    cleaned[rule_rows, :] = 255
    if not bands:
        raise DocumentProcessingError('No text lines were detected in the image. Use a clearer image with separated handwritten lines.')

    if len(bands) > 100:
        raise DocumentProcessingError('The page contains more than 100 detected text lines. Split the document into smaller pages and retry.')
    line_images = []
    layout = []
    for top, bottom in bands:
        band_mask = ink[top:bottom]
        if int(band_mask.sum()) < max(30, int(image.width * 0.08)):
            continue
        if is_rule_only_band(band_mask):
            continue

        # Split a line at a detected ruled column only when that divider
        # actually crosses this text band. This preserves normal sentences
        # elsewhere on the page while keeping table cells apart.
        separators = []
        for start, end, presence in vertical_rule_presence:
            if presence[top:bottom].sum() >= max(2, int((bottom - top) * 0.1)):
                separators.append((start + end) // 2)
        edges = [0, *sorted(set(separators)), image.width]
        for column, (region_left, region_right) in enumerate(zip(edges, edges[1:])):
            region = band_mask[:, region_left:region_right]
            active_columns = np.flatnonzero(region.any(axis=0))
            if not len(active_columns):
                continue
            content_left = region_left + int(active_columns[0])
            content_right = region_left + int(active_columns[-1]) + 1
            padding_x = max(5, round((bottom - top) * 0.15))
            padding_y = max(4, round((bottom - top) * 0.12))
            left = max(region_left, content_left - padding_x)
            right = min(region_right, content_right + padding_x)
            y0 = max(0, top - padding_y)
            y1 = min(image.height, bottom + padding_y)
            crop = Image.fromarray(cleaned[y0:y1, left:right])
            # Keep original grayscale when contrast normalization would collapse
            # a nearly uniform/faint crop; never binarize recognition input.
            if crop.getextrema()[1] - crop.getextrema()[0] < 12:
                crop = gray.crop((left, y0, right, y1))
            crop = ImageOps.autocontrast(crop).convert('RGB')
            crop = ImageOps.expand(crop, border=3, fill='white')
            line_images.append(crop)
            layout.append((top, column, len(edges) - 1))
    if not line_images:
        raise DocumentProcessingError('No handwritten text lines could be prepared for recognition.')
    return (line_images, layout) if with_layout else line_images


def recognize_handwriting(image, language='en'):
    """Return recognized text; no confidence score is exposed because this path is uncalibrated."""
    if language != 'en':
        raise DocumentProcessingError('Only English handwriting is supported. Tamil and Hindi are future extensions.')
    line_images, layout = preprocess_handwriting_image(image, with_layout=True)
    # Validate/segment before loading expensive weights, particularly for blanks.
    torch, processor, model, device_name = _load_backend()
    recognized = []
    try:
        for line_image, (row, column, columns) in zip(line_images, layout):
            pixel_values = processor(images=line_image, return_tensors='pt').pixel_values.to(device_name)
            # Serialize generation within a worker to cap concurrent CPU/GPU memory use.
            with _INFERENCE_LOCK, torch.inference_mode():
                generated_ids = model.generate(pixel_values, max_new_tokens=128, do_sample=False, num_beams=1)
            line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if line_text:
                recognized.append((row, column, columns, line_text))
    except Exception as exc:
        raise DocumentProcessingError('The TrOCR model failed during inference. Try a smaller or clearer page, or check memory availability.') from exc
    rows = {}
    for row, column, columns, text in recognized:
        if row not in rows:
            rows[row] = [''] * columns
        rows[row][column] = text
    table_widths = {len(cells) for cells in rows.values() if sum(bool(cell) for cell in cells) > 1}
    return '\n'.join('\t'.join(cells) if len(cells) in table_widths else ''.join(cells)
                     for cells in rows.values())


def recognize_handwriting_image(image, language='en'):
    """Compatibility entry point used by existing document/resume routes."""
    return recognize_handwriting(image, language=language)


def reset_model_cache():
    """Release cached weights; primarily useful to controlled tests and management tools."""
    global _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = None
