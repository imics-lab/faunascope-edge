import os
import json

import cv2
import numpy as np
from typing import Dict, List

from .speciesnet.constants import Classification, NUM_CLASSIFICATIONS_FOR_ENSEMBLE
from .speciesnet.ensemble_prediction_combiner import combine_predictions
from .speciesnet.geofence_utils import geofence_animal_classification, roll_up_labels_to_first_matching_level

CROP_SIZE = 480
MAX_BATCH = 8
# MegaDetector calls people "person"; the SpeciesNet ensemble calls them "human".
ENSEMBLE_DETECTION_LABELS = {'animal': 'animal', 'person': 'human', 'vehicle': 'vehicle'}


def common_name(label: str) -> str:
    """The last field of a SpeciesNet label, e.g. 'white-tailed deer' or 'cervidae family'."""
    return label.split(';')[-1]


def load_taxonomy(path: str) -> Dict[str, str]:
    """Maps each taxon (class;order;family;genus;species) to its label, as speciesnet's ensemble.py does."""
    def taxa(label):
        return ';'.join(label.split(';')[1:6])
    with open(path, encoding='utf-8') as f:
        taxonomy_map = {taxa(line.strip()): line.strip() for line in f}
    for label in (Classification.BLANK, Classification.VEHICLE, Classification.UNKNOWN):
        taxonomy_map.pop(taxa(label), None)
    for label in (Classification.HUMAN, Classification.ANIMAL):
        taxonomy_map[taxa(label)] = label
    return taxonomy_map


class SpeciesClassifier:
    """
    SpeciesNet's crop classifier (v4.0.3a, 2,498 labels) in the ONNX form written by scripts/export_speciesnet.py, with SpeciesNet's own rules for combining its output with MegaDetector's.

    A track is classified from several of its crops at once: the classifier's probabilities are averaged over the crops, and the five most probable labels go through the SpeciesNet ensemble together with the track's MegaDetector class and confidence. The ensemble accepts a label only above its score thresholds and if the geofence allows it in the given country and state, and otherwise rolls the probabilities up to genus, family, order, class, or kingdom. Because MegaDetector's confidence is at least 0.2 here, the ensemble calls a track blank only when the averaged probability of blank exceeds 0.99.
    """
    def __init__(self, model_dir: str, engine_cache_dir: str, country: str, admin1_region: str):
        import onnxruntime
        available = onnxruntime.get_available_providers()
        providers = []
        if 'TensorrtExecutionProvider' in available:
            os.makedirs(engine_cache_dir, exist_ok=True)
            shape = f'crops:{{}}x{CROP_SIZE}x{CROP_SIZE}x3'
            providers.append(('TensorrtExecutionProvider', {
                'trt_fp16_enable': True, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': engine_cache_dir,
                'trt_profile_min_shapes': shape.format(1), 'trt_profile_opt_shapes': shape.format(MAX_BATCH),
                'trt_profile_max_shapes': shape.format(MAX_BATCH)}))
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        self.session = onnxruntime.InferenceSession(os.path.join(model_dir, 'speciesnet.onnx'), providers=providers)

        with open(os.path.join(model_dir, 'labels.txt'), encoding='utf-8') as f:
            self.labels = [line.strip() for line in f]
        self.taxonomy_map = load_taxonomy(os.path.join(model_dir, 'taxonomy.txt'))
        with open(os.path.join(model_dir, 'geofence.json'), encoding='utf-8') as f:
            self.geofence_map = json.load(f)
        with open(os.path.join(model_dir, 'version.txt')) as f:
            self.version = f.read().strip()
        self.country = country
        self.admin1_region = admin1_region

    @staticmethod
    def crop(frame: np.ndarray, box) -> np.ndarray:
        """The box's region of a BGR frame, as the 480x480 RGB crop the classifier takes."""
        # Rounded as SpeciesNet's own preprocessing rounds: the corner and the size separately.
        x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
        region = frame[y:y + h, x:x + w]
        return cv2.cvtColor(cv2.resize(region, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)

    def classify(self, crops: List[np.ndarray], detector_label: str, detector_confidence: float) -> Dict:
        """Classify one track from its crops."""
        batch = np.stack(crops).astype(np.float32) / 255.0
        logits = np.concatenate([self.session.run(None, {'crops': batch[i:i + MAX_BATCH]})[0]
                                 for i in range(0, len(batch), MAX_BATCH)])
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = (probabilities / probabilities.sum(axis=1, keepdims=True)).mean(axis=0)
        top = np.argsort(-probabilities)[:NUM_CLASSIFICATIONS_FOR_ENSEMBLE]
        classes, scores = [self.labels[i] for i in top], [float(probabilities[i]) for i in top]

        [result] = combine_predictions(
            classifications_list=[{'classes': classes, 'scores': scores}],
            detections=[{'label': ENSEMBLE_DETECTION_LABELS[detector_label], 'conf': detector_confidence}],
            country=self.country, admin1_region=self.admin1_region,
            taxonomy_map=self.taxonomy_map, geofence_map=self.geofence_map, enable_geofence=True,
            geofence_fn=geofence_animal_classification, roll_up_fn=roll_up_labels_to_first_matching_level)
        return {
            'prediction': common_name(result['class']),
            'taxonomy': result['class'],
            'score': round(float(result['score']), 4),
            'source': result['source'],
            'blank': result['class'] == Classification.BLANK,
            'top_classes': [{'label': common_name(c), 'score': round(s, 4)} for c, s in zip(classes, scores)],
        }
