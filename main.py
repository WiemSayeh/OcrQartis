import os
import re
import time
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Singleton EasyOCR ────────────────────────────────────────────────────────
reader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global reader
    logger.info("Chargement EasyOCR... (30s au premier démarrage)")
    try:
        import easyocr
        reader = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
        logger.info("EasyOCR chargé avec succès ✅")
    except Exception as e:
        logger.error(f"Erreur EasyOCR : {e}", exc_info=True)
    yield

app = FastAPI(title="OCR Service - CIN Tunisie", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# CIN tunisien = exactement 8 chiffres
CIN_PATTERN = re.compile(r'(?<!\d)(\d{8})(?!\d)')


def extract_cin_from_lines(lines: list) -> dict:
    if not lines:
        return {"cin": None, "raw_text": "", "confidence": 0.0, "status": "no_text"}

    raw_text      = " ".join(l["text"] for l in lines)
    full_no_space = raw_text.replace(" ", "").replace("-", "").replace(".", "")

    logger.info(f"Texte OCR brut : {raw_text}")

    # Strategie 1 : texte complet concatene
    m = CIN_PATTERN.search(full_no_space)
    if m:
        return {
            "cin":        m.group(1),
            "raw_text":   raw_text,
            "confidence": 1.0,
            "status":     "success",
            "strategy":   "full_text",
        }

    # Strategie 2 : ligne par ligne, haute confiance d'abord
    for l in sorted([l for l in lines if l["score"] > 0.5],
                    key=lambda x: x["score"], reverse=True):
        candidate = l["text"].replace(" ", "").replace("-", "")
        m = CIN_PATTERN.search(candidate)
        if m:
            return {
                "cin":        m.group(1),
                "raw_text":   raw_text,
                "confidence": float(l["score"]),
                "status":     "success",
                "strategy":   "line_by_line",
            }

    # Strategie 3 : sequence de 7-9 chiffres (confiance faible)
    loose = re.search(r'\d{7,9}', full_no_space)
    if loose:
        raw_candidate = loose.group(0)
        cin_candidate = raw_candidate[:8].zfill(8)
        return {
            "cin":        cin_candidate,
            "raw_text":   raw_text,
            "confidence": 0.4,
            "status":     "low_confidence",
            "strategy":   "loose_match",
            "note":       f"Sequence '{raw_candidate}' normalisee en '{cin_candidate}'",
        }

    return {"cin": None, "raw_text": raw_text, "confidence": 0.0, "status": "not_found"}


@app.get("/health")
async def health():
    return {
        "status": "ok" if reader is not None else "loading",
        "model":  "EasyOCR",
        "ready":  reader is not None,
    }


@app.post("/extract-cin")
async def extract_cin(file: UploadFile = File(...)):
    if reader is None:
        raise HTTPException(status_code=503, detail="OCR non pret, reessayez dans quelques secondes.")

    ALLOWED = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/jfif", "image/pjpeg"}
    if file.content_type not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Type non supporte : {file.content_type}")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    suffix = os.path.splitext(file.filename or "img.jpg")[1] or ".jpg"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="cin_ocr_")
    os.close(tmp_fd)

    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        logger.info(f"OCR sur {tmp_path} ({len(contents)} octets)")
        t0 = time.time()

        # EasyOCR retourne : [(bbox, text, score), ...]
        raw_results = reader.readtext(tmp_path, detail=1, paragraph=False)

        elapsed = round(time.time() - t0, 2)
        logger.info(f"OCR termine en {elapsed}s - {len(raw_results)} lignes detectees")

    except Exception as e:
        logger.error(f"Erreur OCR : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur OCR : {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    lines = [{"text": text, "score": float(score)} for (_, text, score) in raw_results]

    result = extract_cin_from_lines(lines)
    result["ocr_time_seconds"] = elapsed
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=False)