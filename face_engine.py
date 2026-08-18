import os
import hashlib
import io
import json
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps

cv2.ocl.setUseOpenCL(False)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass
cv2.setNumThreads(max(1, (os.cpu_count() or 4)))

_CPU = dict(backend_id=cv2.dnn.DNN_BACKEND_OPENCV, target_id=cv2.dnn.DNN_TARGET_CPU)

# ============================= CONFIG =================================
_HERE = os.path.dirname(os.path.abspath(__file__))

YUNET_MODEL = os.path.join(_HERE, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(_HERE, "face_recognition_sface_2021dec.onnx")
DETECT_CONFIDENCE = float(os.environ.get("PICMATCH_DETECT_CONF", 0.5))
NMS_THRESHOLD = 0.3
TOP_K = 5000
TARGET_LONG_SIDE = int(os.environ.get("PICMATCH_TARGET_LONG_SIDE", 1600))
MAX_LONG_SIDE = int(os.environ.get("PICMATCH_MAX_LONG_SIDE", 2200))
MIN_FACE_PX = int(os.environ.get("PICMATCH_MIN_FACE_PX", 32))
COSINE_THRESHOLD = float(os.environ.get("PICMATCH_THRESHOLD", 0.50))
AUTO_ORIENT = os.environ.get("PICMATCH_AUTO_ORIENT", "1") != "0"
_ORIENT_MARGIN = 0.08

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ENGINE_VERSION = "2"
# ========================================================================

_ROTATIONS = (
    (0, None),
    (90, cv2.ROTATE_90_CLOCKWISE),
    (180, cv2.ROTATE_180),
    (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
)

_detector = None
_recognizer = None
_model_lock = threading.Lock()


class ModelError(RuntimeError):
    """Raised when the ONNX model files are missing or unreadable."""


# ------------------------------------------------------------------ models
def ensure_models():
    """Load both ONNX models once, pinned to CPU. Safe to call repeatedly."""
    global _detector, _recognizer
    if _detector is not None and _recognizer is not None:
        return _detector, _recognizer
    with _model_lock:
        if _detector is not None and _recognizer is not None:
            return _detector, _recognizer
        for path in (YUNET_MODEL, SFACE_MODEL):
            if not os.path.isfile(path):
                raise ModelError(
                    f"Missing model file: {os.path.basename(path)}. Download it into "
                    f"{_HERE} - see the setup notes at the top of match_faces.py."
                )
        try:
            _detector = cv2.FaceDetectorYN.create(
                YUNET_MODEL, "", (320, 320),
                DETECT_CONFIDENCE, NMS_THRESHOLD, TOP_K, **_CPU
            )
            _recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "", **_CPU)
        except Exception as exc:
            _detector = _recognizer = None
            raise ModelError(f"Couldn't load the models: {exc}") from exc
    return _detector, _recognizer


# ------------------------------------------------------------------ images
def load_image(source):
    """Read an image from a path or raw bytes into a BGR numpy array.

    Honours EXIF orientation and handles non-ASCII paths, which cv2.imread
    silently fails on under Windows.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            pil = Image.open(io.BytesIO(bytes(source)))
        else:
            with open(source, "rb") as fh:
                pil = Image.open(io.BytesIO(fh.read()))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        return np.array(pil)[:, :, ::-1].copy()
    except Exception:
        try:
            raw = (np.frombuffer(bytes(source), np.uint8)
                   if isinstance(source, (bytes, bytearray))
                   else np.fromfile(source, np.uint8))
            return cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            return None


def _resize_to_working_size(img):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest < TARGET_LONG_SIDE:
        scale = TARGET_LONG_SIDE / longest
    elif longest > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / longest
    else:
        return img
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)


def _detect_raw(img):
    detector, _ = ensure_models()
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or not len(faces):
        return np.empty((0, 15), np.float32)
    keep = np.minimum(faces[:, 2], faces[:, 3]) >= MIN_FACE_PX
    return faces[keep]


def detect_faces(img, auto_orient=None):
    """Detect faces, correcting for photos stored sideways.

    Returns (working_image, faces) where `faces` are YuNet rows valid in the
    coordinate space of the returned `working_image` - so always crop from
    the image this hands back, not from the original.
    """
    if img is None:
        return None, np.empty((0, 15), np.float32)
    work, faces, _, _ = _detect_upright(img, auto_orient)
    return work, faces


def _mean_conf(faces):
    return float(faces[:, -1].mean()) if len(faces) else 0.0


def _detect_upright(img, auto_orient=None):
    if auto_orient is None:
        auto_orient = AUTO_ORIENT

    work = _resize_to_working_size(img)
    faces = _detect_raw(work)
    stats = {0: (len(faces), _mean_conf(faces))}

    if not auto_orient:
        return work, faces, 0, stats
    base_count = len(faces)
    base_score = _mean_conf(faces)

    best_img, best_faces, best_rot = work, faces, 0
    best_score = base_score

    for degrees, code in _ROTATIONS[1:]:
        rotated = cv2.rotate(work, code)
        found = _detect_raw(rotated)
        score = _mean_conf(found)
        stats[degrees] = (len(found), score)
        if len(found) < base_count:
            continue  
        if score > base_score + _ORIENT_MARGIN and score > best_score:
            best_img, best_faces, best_rot, best_score = rotated, found, degrees, score

    return best_img, best_faces, best_rot, stats


def detect_upright_rotation(img):
    if img is None:
        return 0, {}
    _, _, rotation, stats = _detect_upright(img, auto_orient=True)
    return rotation, stats


def rotate_clockwise(img, degrees):
    """Rotate a BGR array by 0/90/180/270 degrees clockwise."""
    code = dict(_ROTATIONS).get(degrees % 360)
    return img if code is None else cv2.rotate(img, code)


# -------------------------------------------------------------- embeddings
def embed_face(img, face):
    """Return a unit-length embedding for one detected face.

    Averages the vector for the face and its mirror image, which makes the
    result a little less sensitive to head pose.
    """
    _, recognizer = ensure_models()
    aligned = recognizer.alignCrop(img, face)
    a = recognizer.feature(aligned).flatten().astype(np.float32)
    a /= np.linalg.norm(a) + 1e-9
    b = recognizer.feature(cv2.flip(aligned, 1)).flatten().astype(np.float32)
    b /= np.linalg.norm(b) + 1e-9
    v = a + b
    return v / (np.linalg.norm(v) + 1e-9)


def embed_all_faces(source):
    img, faces = detect_faces(load_image(source))
    if img is None or not len(faces):
        return []
    return [embed_face(img, f) for f in faces]


def embed_primary_face(source):
    img, faces = detect_faces(load_image(source))
    if img is None or not len(faces):
        return None
    largest = int(np.argmax(faces[:, 2] * faces[:, 3]))
    return embed_face(img, faces[largest])


def similarity(a, b):
    return float(np.dot(np.asarray(a).ravel(), np.asarray(b).ravel()))


def best_similarity(ref, candidates):
    if not len(candidates):
        return 0.0
    return float(np.max(np.asarray(candidates, np.float32) @ np.asarray(ref, np.float32).ravel()))


def is_match(score, threshold=None):
    return score >= (COSINE_THRESHOLD if threshold is None else threshold)


# ------------------------------------------------------------------- misc
def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))


def settings_summary():
    return (f"CPU-only | detect_conf={DETECT_CONFIDENCE} threshold={COSINE_THRESHOLD} "
            f"work={TARGET_LONG_SIDE}-{MAX_LONG_SIDE}px min_face={MIN_FACE_PX}px "
            f"auto_orient={'on' if AUTO_ORIENT else 'off'}")


# ------------------------------------------------------------ embedding cache
class EmbeddingCache:

    def __init__(self, path):
        self.path = path
        self._data = {}
        self._lock = threading.Lock()
        self.load()

    @staticmethod
    def _stamp(file_path):
        st = os.stat(file_path)
        key = f"{ENGINE_VERSION}|{DETECT_CONFIDENCE}|{TARGET_LONG_SIDE}|{MAX_LONG_SIDE}|" \
              f"{MIN_FACE_PX}|{AUTO_ORIENT}|{st.st_size}|{int(st.st_mtime)}"
        return hashlib.sha1(key.encode()).hexdigest()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._data = {
                k: {"stamp": v["stamp"],
                    "embeddings": [np.asarray(e, np.float32) for e in v["embeddings"]]}
                for k, v in raw.items()
            }
        except Exception:
            self._data = {}

    def save(self):
        try:
            with self._lock:
                raw = {k: {"stamp": v["stamp"],
                           "embeddings": [e.tolist() for e in v["embeddings"]]}
                       for k, v in self._data.items()}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(raw, fh)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def get(self, name, file_path):
        try:
            stamp = self._stamp(file_path)
        except OSError:
            return []
        with self._lock:
            hit = self._data.get(name)
            if hit is not None and hit["stamp"] == stamp:
                return hit["embeddings"]
        embeds = embed_all_faces(file_path)
        with self._lock:
            self._data[name] = {"stamp": stamp, "embeddings": embeds}
        return embeds

    def drop(self, name):
        with self._lock:
            self._data.pop(name, None)

    def prune(self, keep_names):
        keep = set(keep_names)
        with self._lock:
            for gone in [k for k in self._data if k not in keep]:
                del self._data[gone]
