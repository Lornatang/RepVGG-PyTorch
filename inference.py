# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os

import torch
from torch import nn

import model
from imgproc import preprocess_one_image
from utils import load_class_label, load_pretrained_state_dict


def main():
    # Get the label name corresponding to the drawing
    class_label_map = load_class_label(args.class_label_file, args.model_num_classes)
    # Define the running device
    device = torch.device(args.device)

    # read image
    tensor = preprocess_one_image(args.image_path,
                                  args.image_size,
                                  args.model_mean_parameters,
                                  args.model_std_parameters,
                                  False,
                                  args.half,
                                  device)

    # Initialize the model
    cls_model = build_model(args.model_arch_name,
                            args.model_num_classes,
                            args.inference_mode,
                            args.use_checkpoint,
                            args.half, device)
    print(f"Build `{args.model_arch_name}` model successfully.")
    cls_model = load_pretrained_state_dict(cls_model, False, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Inference
    with torch.no_grad():
        output = cls_model(tensor)

    # Calculate the five categories with the highest classification probability
    prediction_class_index = torch.topk(output, k=5).indices.squeeze(0).tolist()

    # Print classification results
    for class_index in prediction_class_index:
        prediction_class_label = class_label_map[class_index]
        prediction_class_prob = torch.softmax(output, dim=1)[0, class_index].item()
        print(f"{prediction_class_label:<75} ({prediction_class_prob * 100:.2f}%)")


def build_model(
        model_arch_name: str,
        model_num_classes: int,
        inference_mode: bool,
        use_checkpoint: bool,
        half: bool,
        device: torch.device,
) -> nn.Module:
    # Define the model structure according to the model name
    cls_model = model.__dict__[model_arch_name](num_classes=model_num_classes,
                                                inference_mode=inference_mode,
                                                use_checkpoint=use_checkpoint)

    # If half-precision floating-point calculation is enabled, the model needs to be converted to a half-precision floating-point model
    if half:
        cls_model.half()

    # Move the model to the specified device
    cls_model = cls_model.to(device)

    # Set the model to evaluation mode
    cls_model.eval()

    return cls_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./figure/n01440764_36.JPEG")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--model_mean_parameters", type=list, default=[0.485, 0.456, 0.406])
    parser.add_argument("--model_std_parameters", type=list, default=[0.229, 0.224, 0.225])
    parser.add_argument("--model_arch_name", type=str, default="reg_vgg_a0")
    parser.add_argument("--model_num_classes", type=int, default=1000)
    parser.add_argument("--inference_mode", type=bool, default=False)
    parser.add_argument("--use_checkpoint", type=bool, default=False)
    parser.add_argument("--class_label_file", type=str, default="./data/ImageNet_1K_labels_map.txt")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/pretrained_models/RepVGG_A0-ImageNet_1K.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--half",
                        action="store_true",
                        help="Use half precision.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="Device to run model.")
    args = parser.parse_args()

    main()
