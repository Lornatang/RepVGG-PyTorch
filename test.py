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
import time
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import model
from dataset import CUDAPrefetcher, ImageDataset
from utils import accuracy, load_pretrained_state_dict, AverageMeter, ProgressMeter, Summary


def load_dataset(config: Any, device: torch.device) -> CUDAPrefetcher:
    test_datasets = ImageDataset(config["TEST"]["DATASET"]["TEST_IMAGES_DIR"],
                                 config["TEST"]["DATASET"]["TEST_IMAGES_SIZE"],
                                 config["DATASET_MEAN_PARAMETERS"],
                                 config["DATASET_STD_PARAMETERS"],
                                 "Test")
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                 num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                 drop_last=False,
                                 persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])

    test_data_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return test_data_prefetcher


def build_model(
        config: Any,
        device: torch.device,
) -> nn.Module:
    cls_model = model.__dict__[config["MODEL"]["CLS"]["NAME"]](num_classes=config["MODEL"]["CLS"]["NUM_CLASSES"],
                                                               inference_mode=config["MODEL"]["CLS"]["INFERENCE_MODE"],
                                                               use_checkpoint=config["MODEL"]["CLS"]["USE_CHECKPOINT"])
    cls_model = cls_model.to(device)

    # compile model
    if config["MODEL"]["CLS"]["COMPILED"]:
        cls_model = torch.compile(cls_model)

    return cls_model


def test(
        cls_model: nn.Module,
        test_data_prefetcher: CUDAPrefetcher,
        device: torch.device,
) -> [float, float]:
    # Calculate the number of iterations per epoch
    batches = len(test_data_prefetcher)
    # Interval printing
    if batches > 100:
        print_freq = 100
    else:
        print_freq = batches
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(len(test_data_prefetcher),
                             [batch_time, data_time, acc1, acc5],
                             prefix="Test: ")

    # set the model as validation model
    cls_model.eval()

    with torch.no_grad():
        # Initialize data batches
        batch_index = 0

        # Set the data set iterator pointer to 0 and load the first batch of data
        test_data_prefetcher.reset()
        batch_data = test_data_prefetcher.next()

        # Record the start time of verifying a batch
        end = time.time()

        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["images"].to(device, non_blocking=True)
            target = batch_data["target"].to(device, non_blocking=True)

            # Record the time to load a batch of data
            data_time.update(time.time() - end)

            output = cls_model(images)

            # record current metrics
            # Get batch size
            batch_size = images.size(0)
            top1, top5 = accuracy(output, target, topk=(1, 5))
            acc1.update(top1[0].item(), batch_size)
            acc5.update(top5[0].item(), batch_size)

            # Record the total time to verify a batch
            batch_time.update(time.time() - end)
            end = time.time()

            # Output a verification log information
            if batch_index % print_freq == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = test_data_prefetcher.next()

            # Add 1 to the number of data batches
            batch_index += 1

    # Print the performance index of the model at the current Epoch
    progress.display_summary()

    return acc1.avg, acc5.avg


def main() -> None:
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/test/REPVGG_A0_plain.yaml",
                        required=True,
                        help="Path to test config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    device = torch.device("cuda", config["DEVICE_ID"])
    test_data_prefetcher = load_dataset(config, device)
    cls_model = build_model(config, device)

    # Load model weights
    cls_model = load_pretrained_state_dict(cls_model, config["MODEL"]["CLS"]["COMPILED"], config["MODEL_WEIGHTS_PATH"])

    test(cls_model,
         test_data_prefetcher,
         device)


if __name__ == "__main__":
    main()
