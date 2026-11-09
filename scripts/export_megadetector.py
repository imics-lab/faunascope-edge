"""
Converts MegaDetector V6 (MDV6-mit-yolov9-c, from the MIT-licensed YOLO code) to ONNX, the format the edge detector loads.

The checkpoint and its config come from https://zenodo.org/records/15398270 (MDV6-mit-yolov9-c.ckpt, config_v9s.yaml). The model code is the copy of MultimediaTechLab's YOLO that Pytorch-Wildlife 1.3.1 ships in PytorchWildlife/models/detection/yolo_mit/; pass that folder as --yolo-code. The script needs torch, torchvision, omegaconf, rich, and onnx, but not the rest of Pytorch-Wildlife.

The ONNX model takes a 1x3x640x640 RGB image scaled to [0, 1] and returns boxes (1xNx4, x1 y1 x2 y2 in input pixels) and scores (1xNx3: animal, person, vehicle, after the sigmoid). Non-maximum suppression is left to the caller.

    python scripts/export_megadetector.py --yolo-code <...>/yolo_mit --weights MDV6-mit-yolov9-c.ckpt --config config_v9s.yaml --output model_weight/MDV6-mit-yolov9-c.onnx
"""
import argparse
import sys

import torch
import yaml
from omegaconf import OmegaConf

IMAGE_SIZE = 640


class Exportable(torch.nn.Module):
    def __init__(self, model, converter):
        super().__init__()
        self.model = model
        self.converter = converter

    def forward(self, image):
        scores, _, boxes = self.converter(self.model(image)["Main"])
        return boxes, scores.sigmoid()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--yolo-code', required=True, help="Folder that contains the 'yolo' package")
    parser.add_argument('--weights', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.yolo_code)
    from yolo import create_model, create_converter

    with open(args.config) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    cfg.image_size = [IMAGE_SIZE, IMAGE_SIZE]
    model = create_model(cfg.model, weight_path=args.weights, class_num=3).eval()
    converter = create_converter(cfg.model.name, model, cfg.model.anchor, cfg.image_size, 'cpu')

    torch.onnx.export(Exportable(model, converter), torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE), args.output,
                      input_names=['image'], output_names=['boxes', 'scores'], opset_version=17, dynamo=False)
    print(f"Wrote {args.output}")


if __name__ == '__main__':
    main()
