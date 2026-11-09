"""
Converts Google's SpeciesNet classifier to ONNX and gathers the files the edge classifier loads.

Download the PyTorch release (no Kaggle account is needed) and pass its folder as --source:

    python -c "import kagglehub; print(kagglehub.model_download('google/speciesnet/pyTorch/v4.0.3a/1'))"
    python scripts/export_speciesnet.py --source <that folder> --output model_weight/speciesnet

The script needs torch, onnx2torch (the release is a model converted with it), and onnx. It writes speciesnet.onnx (input: a batch of 480x480 RGB crops, NHWC, scaled to [0, 1]; output: logits), labels.txt, taxonomy.txt, geofence.json, and version.txt. The model is Apache-2.0 licensed.
"""
import argparse
import json
import os
import shutil

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--source', required=True, help="Folder of the SpeciesNet PyTorch release (contains info.json)")
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.source, 'info.json')) as f:
        info = json.load(f)
    if info['type'] != 'always_crop':
        raise SystemExit(f"Expected the crop classifier (v4.0.*a), got type {info['type']}")
    os.makedirs(args.output, exist_ok=True)

    model = torch.load(os.path.join(args.source, info['classifier']), map_location='cpu', weights_only=False).eval()
    torch.onnx.export(model, torch.zeros(1, 480, 480, 3), os.path.join(args.output, 'speciesnet.onnx'),
                      input_names=['crops'], output_names=['logits'],
                      dynamic_axes={'crops': {0: 'n'}, 'logits': {0: 'n'}}, opset_version=17, dynamo=False)
    for key, name in (('classifier_labels', 'labels.txt'), ('taxonomy', 'taxonomy.txt'), ('geofence', 'geofence.json')):
        shutil.copyfile(os.path.join(args.source, info[key]), os.path.join(args.output, name))
    with open(os.path.join(args.output, 'version.txt'), 'w') as f:
        f.write(info['version'] + '\n')
    print(f"Wrote SpeciesNet {info['version']} to {args.output}")


if __name__ == '__main__':
    main()
