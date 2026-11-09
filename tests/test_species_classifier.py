"""SpeciesClassifier's averaging and the SpeciesNet ensemble rules, with a stand-in for the ONNX session and a small label set."""
import json

import numpy as np
import pytest

from src.animal_analyzer.species_classifier import SpeciesClassifier, load_taxonomy

LABELS = [
    "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank",
    "1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal",
    "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human",
    "5c7ce479-8a45-40b3-ae21-7c97dfae22f5;mammalia;artiodactyla;cervidae;odocoileus;virginianus;white-tailed deer",
    "d3b1b4e0-0000-0000-0000-000000000001;mammalia;artiodactyla;cervidae;odocoileus;hemionus;mule deer",
    "d3b1b4e0-0000-0000-0000-000000000002;mammalia;artiodactyla;cervidae;;;cervidae family",
    "d3b1b4e0-0000-0000-0000-000000000003;amphibia;anura;bufonidae;rhinella;marina;cane toad",
    "d3b1b4e0-0000-0000-0000-000000000004;amphibia;anura;bufonidae;;;bufonidae family",
]
BLANK, ANIMAL, HUMAN, WTD, MULE, CERVIDAE, TOAD, BUFONIDAE = range(len(LABELS))


class FakeSession:
    """Returns one row of logits per crop: crop k gets logits_per_crop[k]."""

    def __init__(self, logits_per_crop):
        self.logits = np.array(logits_per_crop, np.float32)
        self.batches = []

    def run(self, _, feeds):
        n = len(feeds["crops"])
        start = sum(self.batches)
        self.batches.append(n)
        return [self.logits[start:start + n]]


@pytest.fixture
def make_classifier(tmp_path):
    (tmp_path / "taxonomy.txt").write_text("\n".join(LABELS) + "\n")
    geofence = {"amphibia;anura;bufonidae;rhinella;marina": {"allow": {"USA": ["FL", "HI"]}}}

    def make(logits_per_crop):
        classifier = SpeciesClassifier.__new__(SpeciesClassifier)
        classifier.session = FakeSession(logits_per_crop)
        classifier.labels = LABELS
        classifier.taxonomy_map = load_taxonomy(str(tmp_path / "taxonomy.txt"))
        classifier.geofence_map = json.loads(json.dumps(geofence))
        classifier.country, classifier.admin1_region = "USA", "TX"
        return classifier
    return make


def logits(**scores):
    """Logits whose softmax puts the given probabilities on the named labels and spreads the rest evenly."""
    p = np.full(len(LABELS), (1 - sum(scores.values())) / (len(LABELS) - len(scores)))
    for name, value in scores.items():
        p[globals()[name]] = value
    return np.log(p)


def crops(n):
    return [np.zeros((480, 480, 3), np.uint8)] * n


def test_a_confident_species_is_named(make_classifier):
    result = make_classifier([logits(WTD=0.95)]).classify(crops(1), "animal", 0.9)
    assert (result["prediction"], result["blank"]) == ("white-tailed deer", False)
    assert result["score"] == pytest.approx(0.95, abs=1e-3)
    assert result["top_classes"][0] == {"label": "white-tailed deer", "score": pytest.approx(0.95, abs=1e-3)}


def test_probabilities_are_averaged_over_the_crops(make_classifier):
    # Each crop alone would name its deer; averaged, the two split the vote and roll up to the family (this taxonomy has no genus entry).
    classifier = make_classifier([logits(WTD=0.9), logits(MULE=0.9)])
    result = classifier.classify(crops(2), "animal", 0.9)
    assert (result["prediction"], result["source"]) == ("cervidae family", "classifier+rollup_to_family")


def test_crops_are_sent_in_batches_of_at_most_eight(make_classifier):
    classifier = make_classifier([logits(WTD=0.95)] * 10)
    classifier.classify(crops(10), "animal", 0.9)
    assert classifier.session.batches == [8, 2]


def test_a_species_outside_its_geofence_is_rolled_up(make_classifier):
    result = make_classifier([logits(TOAD=0.95)]).classify(crops(1), "animal", 0.9)
    assert result["prediction"] == "bufonidae family"


def test_only_a_very_confident_blank_is_blank(make_classifier):
    assert make_classifier([logits(BLANK=0.995)]).classify(crops(1), "animal", 0.3)["blank"]
    assert not make_classifier([logits(BLANK=0.95)]).classify(crops(1), "animal", 0.3)["blank"]


def test_a_confident_person_detection_is_human(make_classifier):
    result = make_classifier([logits(WTD=0.9)]).classify(crops(1), "person", 0.8)
    assert (result["prediction"], result["source"]) == ("human", "detector")
