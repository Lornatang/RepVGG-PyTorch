# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
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
from model import convert_inference_model
from utils import load_pretrained_state_dict
from inference import build_model
import torch
import argparse
import os


def main():
    # Define the running device
    device = torch.device(args.device)

    # Initialize the model
    cls_model = build_model(args.model_arch_name,
                            args.model_num_classes,
                            False,
                            True,
                            args.half,
                            device)
    print(f"Build `{args.model_arch_name}` model successfully.")
    cls_model = load_pretrained_state_dict(cls_model, False, args.inputs_model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.inputs_model_weights_path)}` successfully.")

    # Convert to inference model
    convert_inference_model(cls_model, args.output_model_weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch_name", type=str, default="reg_vgg_a0")
    parser.add_argument("--model_num_classes", type=int, default=1000)
    parser.add_argument("-i", "--inputs_model_weights_path",
                        type=str,
                        default="./results/pretrained_models/RepVGG_A0-ImageNet_1K.pth.tar",
                        help="Input model weights file path.")
    parser.add_argument("-o", "--output_model_weights_path",
                        type=str,
                        default="./results/pretrained_models/RepVGG_A0_plain-ImageNet_1K.pth.tar",
                        help="Output model weights file path.")
    parser.add_argument("--half",
                        action="store_true",
                        help="Use half precision.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="Device to run model.")
    args = parser.parse_args()

    main()
