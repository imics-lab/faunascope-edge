# Models

The device runs two published models, both converted to ONNX so that the same files run under ONNX Runtime on a Jetson (with TensorRT), a desktop GPU, or a CPU. The device does not train or fine-tune anything.

| | MegaDetector V6 | SpeciesNet |
|---|---|---|
| Task | Find animals, people, and vehicles | Name the species of a cropped animal |
| Version | `MDV6-mit-yolov9-c` | 4.0.3a (crop classifier), label release 2026-06-09 |
| Authors | Microsoft AI for Good (Pytorch-Wildlife) | Google |
| Architecture | YOLOv9 (MultimediaTechLab's MIT-licensed implementation), 9.7 M parameters, 640 x 640 input | EfficientNetV2-M, 52.7 M parameters, 480 x 480 input |
| License | Code MIT; weights CC BY 4.0 | Apache-2.0 |
| File on the device | `model_weight/MDV6-mit-yolov9-c.onnx` (29 MB, in the repository) | `model_weight/speciesnet/` (224 MB, downloaded from this repository's releases) |
| Converted by | `scripts/export_megadetector.py` | `scripts/export_speciesnet.py` |

## MegaDetector

MegaDetector V6 comes in several variants. `MDV6-mit-yolov9-c` is the compact model trained with MultimediaTechLab's MIT-licensed YOLO code, which keeps the whole edge pipeline free of copyleft licenses; the other compact variant, `MDV6-yolov9-c`, was trained with and runs on Ultralytics' AGPL-licensed code.

The checkpoint and its model configuration are on Zenodo (record 15398270: `MDV6-mit-yolov9-c.ckpt` and `config_v9s.yaml`). The model code is the copy of the YOLO package that Pytorch-Wildlife 1.3.1 ships in `PytorchWildlife/models/detection/yolo_mit/`. To rebuild the ONNX file:

```sh
pip download --no-deps --no-binary :all: PytorchWildlife==1.3.1   # then unpack it
python scripts/export_megadetector.py --yolo-code <unpacked>/src/PytorchWildlife/models/detection/yolo_mit \
    --weights MDV6-mit-yolov9-c.ckpt --config config_v9s.yaml --output model_weight/MDV6-mit-yolov9-c.onnx
```

The script needs `torch`, `torchvision`, `omegaconf`, `rich`, and `onnx`. The exported model includes the conversion of the network's outputs to boxes; it takes a 1 x 3 x 640 x 640 RGB image scaled to [0, 1] and returns boxes and per-class scores, and the device applies the confidence threshold and non-maximum suppression itself. As in Pytorch-Wildlife, the checkpoint's regular weights are used, not its exponential moving average.

## SpeciesNet

SpeciesNet's crop classifier (the "a" release; the "b" release classifies whole images) is published on Kaggle as `google/speciesnet/pyTorch/v4.0.3a/1`, which needs no account to download. The release contains the classifier, its 2,498 labels, the taxonomy used to roll labels up, and the geofence that lists where each species occurs. To rebuild the files:

```sh
python -c "import kagglehub; print(kagglehub.model_download('google/speciesnet/pyTorch/v4.0.3a/1'))"
python scripts/export_speciesnet.py --source <that folder> --output model_weight/speciesnet
```

The script needs `torch`, `onnx2torch` (the PyTorch release is itself a converted model), and `onnx`, and writes `speciesnet.onnx` (input: a batch of 480 x 480 RGB crops, channels last, scaled to [0, 1]; output: logits), `labels.txt`, `taxonomy.txt`, `geofence.json`, and `version.txt`. Because `speciesnet.onnx` is larger than GitHub's 100 MB file limit, these files are published as the release `speciesnet-v4.0.3a` of this repository; the [Deployment](deployment.md#model-weights) guide shows how to download them.

The rules that combine SpeciesNet's output with MegaDetector's (score thresholds, geofencing, taxonomic roll-up) are copied unchanged from Google's `speciesnet` package 5.0.6 into `src/animal_analyzer/speciesnet/`, apart from relative imports. The package itself is not installed on the device, because it brings dependencies the device does not need.

## Running the models

ONNX Runtime runs each model with the first available of its TensorRT, CUDA, and CPU execution providers. The Jetson image (`dustynv/l4t-ml:r36.4.0`) already contains ONNX Runtime 1.22 with TensorRT 10.3. With TensorRT, the models run in FP16, and the first start builds an engine for each, about 8 minutes for MegaDetector and 11 for SpeciesNet on the Orin Nano, kept in `data/engine_cache/` for later starts. The engines are built for the device's GPU and TensorRT version; a changed model gets a new engine automatically.

## Validation

- **Conversion.** The ONNX models give the same outputs as the PyTorch models: MegaDetector scores within 2e-6, SpeciesNet logits within 1.3e-5.
- **MegaDetector on the device versus Pytorch-Wildlife.** On 37 archive frames at the camera's 640 x 480 resolution, the boxes and confidences agree exactly. On 83 frames at 1280 x 720, which must be shrunk, the two differ only in detections close to the 0.2 threshold, because Pytorch-Wildlife resizes with PIL's Lanczos filter and the device with OpenCV's area averaging.
- **SpeciesNet crops versus SpeciesNet's own preprocessing.** On 130 animal crops from archive frames, the top label agrees in 129, with a median difference in probability of 0.001.
- **FP16 on the Jetson versus FP32.** On 120 archive frames, MegaDetector's FP16 engine finds the same boxes, with confidences within 0.04. Calls close to SpeciesNet's thresholds can differ between machines: one grey fox scored 0.68 on a CPU, just above the 0.65 limit, and became "animal" on a desktop GPU.
- **First results on archive videos.** Run over 21 archived events the old YOLO model had labeled "Fox", the two models named a grey fox in 7, a canine in 3, only "animal" or "mammal" in 7 (mostly dark infrared clips), and another animal in 3; MegaDetector did not keep 1. Of 21 events labeled "Mouse_Rat", MegaDetector kept 2; the rest are most likely false triggers of the old model at a water trough at night.
