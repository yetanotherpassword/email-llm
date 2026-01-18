"""YOLO-based fast image filtering and object detection."""

import json
import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from ultralytics import YOLO

from .config import settings
from .models import ImageAnalysis, YoloDetection

logger = logging.getLogger(__name__)

# Object categories for filtering
FACE_CLASSES = {"person"}  # YOLO detects persons, we can use face detection separately
CHART_CLASSES = {"tv", "laptop", "cell phone", "book"}  # Proxies for screens/documents
TEXT_INDICATOR_CLASSES = {"book", "laptop", "tv", "cell phone"}

# Common COCO classes that YOLO can detect
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


class YoloAnalyzer:
    """Fast image analysis using YOLO for object detection."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize YOLO model."""
        self.model_path = model_path or settings.yolo_model
        self.confidence_threshold = settings.yolo_confidence_threshold

        # Cache directory for results
        self.cache_dir = settings.get_yolo_cache_path()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model lazily
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        """Lazy load the YOLO model."""
        if self._model is None:
            logger.info(f"Loading YOLO model: {self.model_path}")
            self._model = YOLO(self.model_path)
        return self._model

    def analyze_image(self, img: Image.Image, cache_key: Optional[str] = None) -> ImageAnalysis:
        """Analyze an image using YOLO.

        Args:
            img: PIL Image to analyze
            cache_key: Optional key for caching results

        Returns:
            ImageAnalysis with detection results
        """
        # Check cache first
        if cache_key:
            cached = self._load_from_cache(cache_key)
            if cached:
                return cached

        # Run YOLO detection
        results = self.model(img, verbose=False, conf=self.confidence_threshold)

        detections = []
        detected_classes = set()

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = tuple(box.xyxy[0].tolist())

                    detections.append(
                        YoloDetection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                        )
                    )
                    detected_classes.add(class_name)

        # Analyze what was found
        has_faces = bool(FACE_CLASSES & detected_classes)
        has_text = bool(TEXT_INDICATOR_CLASSES & detected_classes)
        has_charts = bool(CHART_CLASSES & detected_classes)

        analysis = ImageAnalysis(
            yolo_detections=detections,
            has_faces=has_faces,
            has_text=has_text,
            has_charts=has_charts,
            detected_objects=list(detected_classes),
        )

        # Cache results
        if cache_key:
            self._save_to_cache(cache_key, analysis)

        return analysis

    def filter_images_by_object(
        self,
        images: list[tuple[str, Image.Image]],
        object_class: str,
    ) -> list[str]:
        """Filter images that contain a specific object class.

        Args:
            images: List of (id, image) tuples
            object_class: Object class to search for

        Returns:
            List of image IDs that contain the object
        """
        matching_ids = []

        for img_id, img in images:
            analysis = self.analyze_image(img, cache_key=img_id)
            if object_class.lower() in [obj.lower() for obj in analysis.detected_objects]:
                matching_ids.append(img_id)

        return matching_ids

    def filter_images_with_faces(
        self,
        images: list[tuple[str, Image.Image]],
    ) -> list[str]:
        """Filter images that contain people/faces.

        Args:
            images: List of (id, image) tuples

        Returns:
            List of image IDs that contain faces
        """
        matching_ids = []

        for img_id, img in images:
            analysis = self.analyze_image(img, cache_key=img_id)
            if analysis.has_faces:
                matching_ids.append(img_id)

        return matching_ids

    def get_detection_summary(self, analysis: ImageAnalysis) -> str:
        """Generate a text summary of YOLO detections for embedding."""
        if not analysis.yolo_detections:
            return "No objects detected in image."

        # Count objects by class
        counts = {}
        for det in analysis.yolo_detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1

        parts = []
        for class_name, count in sorted(counts.items()):
            if count == 1:
                parts.append(class_name)
            else:
                parts.append(f"{count} {class_name}s")

        summary = f"Image contains: {', '.join(parts)}."

        if analysis.has_faces:
            summary += " Contains people."
        if analysis.has_text:
            summary += " May contain text or documents."
        if analysis.has_charts:
            summary += " May contain charts or screens."

        return summary

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[ImageAnalysis]:
        """Load analysis results from cache."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                return ImageAnalysis(**data)
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_key}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, analysis: ImageAnalysis) -> None:
        """Save analysis results to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            cache_path.write_text(analysis.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")


def get_available_object_classes() -> list[str]:
    """Return list of object classes YOLO can detect."""
    return COCO_CLASSES.copy()
