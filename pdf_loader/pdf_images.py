"""Utilities for extracting images and rendered page bitmaps from PDF pages."""

from pathlib import Path
from typing import Dict, List, Optional
import shutil
import fitz  # PyMuPDF


def extract_images(
    pdf_path: Path,
    image_dir: Path,
    *,
    page_numbers: Optional[List[int]] = None,
    render_pages_dpi: Optional[int] = None,
    diagnostic_dir: Optional[Path] = None,
) -> Dict[int, List[Path]]:
    """Extract embedded images and optionally full-page renderings.

    Returns mapping of page number to list of image paths.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        # Determine pages
        total_pages = len(doc)
        if page_numbers:
            pages = sorted({p for p in page_numbers if 1 <= p <= total_pages})
        else:
            pages = list(range(1, total_pages + 1))

        page_images: Dict[int, List[Path]] = {}
        total = len(pages)

        for idx, pnum in enumerate(pages, start=1):
            try:
                print(f"[pdf-images] {idx}/{total} (p={pnum})", flush=True)
            except Exception:
                pass
            page = doc[pnum - 1]
            page_image_paths: List[Path] = []

            # 1) Extract embedded images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                # Prefer robust byte extraction to avoid colorspace issues
                try:
                    info = doc.extract_image(xref)
                    data = info.get("image") if info else None
                    if data:
                        ext = (info.get("ext") or "png").lower()
                        img_filename = f"page_{pnum:04d}_img_{img_index + 1}.{ext}"
                        img_path = image_dir / img_filename
                        img_path.write_bytes(data)
                        page_image_paths.append(img_path)
                        continue
                except Exception:
                    pass

                # Fallback: render via Pixmap
                try:
                    pix = fitz.Pixmap(doc, xref)
                    # Convert to RGB when colorspace present; some masks have None colorspace
                    if getattr(pix, "colorspace", None) is not None:
                        if pix.n - pix.alpha < 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_filename = f"page_{pnum:04d}_img_{img_index + 1}.png"
                    img_path = image_dir / img_filename
                    pix.save(str(img_path))
                    page_image_paths.append(img_path)
                except Exception as e:
                    # Downgrade to warning; full-page render still provides OCR coverage
                    print(f"Warning: fallback image extract failed for page {pnum} img {img_index}: {e}")
                finally:
                    try:
                        if 'pix' in locals():
                            pix = None
                    except Exception:
                        pass

            # 2) Optional full-page render for OCR
            if render_pages_dpi and render_pages_dpi > 0:
                try:
                    zoom = render_pages_dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    render_path = image_dir / f"page_{pnum:04d}_render_{render_pages_dpi}dpi.png"
                    pix.save(str(render_path))
                    page_image_paths.append(render_path)
                    if diagnostic_dir is not None:
                        diag_path = diagnostic_dir / f"page_{pnum:04d}_render_{render_pages_dpi}dpi.png"
                        try:
                            shutil.copyfile(render_path, diag_path)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Error rendering page {pnum} at {render_pages_dpi} DPI: {e}")
                finally:
                    try:
                        if 'pix' in locals():
                            pix = None
                    except Exception:
                        pass

            if page_image_paths:
                page_images[pnum] = page_image_paths

        return page_images
    finally:
        doc.close()
