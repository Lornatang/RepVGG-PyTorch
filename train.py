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
import random
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
from dataset import CUDAPrefetcher, ImageDataset
from test import test
from utils import accuracy, load_resume_state_dict, load_pretrained_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter


def main():
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/train/REPVGG_A0.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    # Fixed random number seed
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Default to start training from scratch
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_acc1 = 0.0

    # Define the running device number
    device = torch.device("cuda", config["DEVICE_ID"])

    # Define the basic functions needed to start training
    train_data_prefetcher, valid_data_prefetcher = load_dataset(config, device)
    cls_model, ema_cls_model = build_model(config, device)
    criterion = define_loss(config, device)
    optimizer = define_optimizer(config, cls_model)
    scheduler = define_scheduler(config, optimizer)

    # Load the pretrained model
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_MODEL"]:
        cls_model = load_pretrained_state_dict(cls_model,
                                               config["MODEL"]["CLS"]["COMPILED"],
                                               config["TRAIN"]["CHECKPOINT"]["PRETRAINED_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    # Load the last training interruption model node
    if config["TRAIN"]["CHECKPOINT"]["RESUMED_MODEL"]:
        cls_model, ema_cls_model, start_epoch, best_acc1, optimizer, scheduler = load_resume_state_dict(
            cls_model,
            ema_cls_model,
            optimizer,
            scheduler,
            config["MODEL"]["CLS"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUMED_MODEL"],
        )
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['RESUMED_MODEL']}` resume model weights successfully.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create the folder where the model weights are saved
    samples_dir = os.path.join("samples", config["EXP_NAME"])
    results_dir = os.path.join("results", config["EXP_NAME"])
    make_directory(samples_dir)
    make_directory(results_dir)

    # create model training log
    writer = SummaryWriter(os.path.join("samples", "logs", config["EXP_NAME"]))

    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train(cls_model,
              ema_cls_model,
              train_data_prefetcher,
              criterion,
              optimizer,
              epoch,
              scaler,
              writer,
              device,
              config)

        # Update LR
        scheduler.step()

        acc1, _ = test(cls_model, valid_data_prefetcher, device)
        print("\n")

        # Write the evaluation indicators of each round of Epoch to the log
        writer.add_scalar(f"Test/Acc@1", acc1, epoch + 1)

        # Automatically save the model with the highest index
        is_best = acc1 > best_acc1
        is_last = (epoch + 1) == config["TRAIN"]["HYP"]["EPOCHS"]
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({"epoch": epoch + 1,
                         "best_acc1": best_acc1,
                         "state_dict": cls_model.state_dict(),
                         "ema_state_dict": ema_cls_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset(
        config: Any,
        device: torch.device,
) -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = ImageDataset(config["TRAIN"]["DATASET"]["TRAIN_IMAGES_DIR"],
                                  config["TRAIN"]["DATASET"]["TRAIN_IMAGES_SIZE"],
                                  config["DATASET_MEAN_PARAMETERS"],
                                  config["DATASET_STD_PARAMETERS"],
                                  "Train")
    valid_datasets = ImageDataset(config["VALID"]["DATASET"]["VALID_IMAGES_DIR"],
                                  config["VALID"]["DATASET"]["VALID_IMAGES_SIZE"],
                                  config["DATASET_MEAN_PARAMETERS"],
                                  config["DATASET_STD_PARAMETERS"],
                                  "Valid")

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                  shuffle=config["TRAIN"]["HYP"]["SHUFFLE"],
                                  num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                  pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                  drop_last=True,
                                  persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"])

    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=config["VALID"]["HYP"]["IMGS_PER_BATCH"],
                                  shuffle=config["VALID"]["HYP"]["SHUFFLE"],
                                  num_workers=config["VALID"]["HYP"]["NUM_WORKERS"],
                                  pin_memory=config["VALID"]["HYP"]["PIN_MEMORY"],
                                  drop_last=False,
                                  persistent_workers=config["VALID"]["HYP"]["PERSISTENT_WORKERS"])

    # Place all data on the preprocessing data loader
    train_data_prefetcher = CUDAPrefetcher(train_dataloader, device)
    valid_data_prefetcher = CUDAPrefetcher(valid_dataloader, device)

    return train_data_prefetcher, valid_data_prefetcher


def build_model(
        config: Any,
        device: torch.device,
) -> [nn.Module, nn.Module or Any]:
    cls_model = model.__dict__[config["MODEL"]["CLS"]["NAME"]](num_classes=config["MODEL"]["CLS"]["NUM_CLASSES"],
                                                               inference_mode=config["MODEL"]["CLS"]["INFERENCE_MODE"],
                                                               use_checkpoint=config["MODEL"]["CLS"]["USE_CHECKPOINT"])
    cls_model = cls_model.to(device)

    if config["MODEL"]["EMA"]["ENABLE"]:
        # Generate an exponential average model based on a generator to stabilize model training
        ema_decay = config["MODEL"]["EMA"]["DECAY"]
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
        ema_cls_model = AveragedModel(cls_model, device=device, avg_fn=ema_avg_fn)
    else:
        ema_cls_model = None

    # compile model
    if config["MODEL"]["CLS"]["COMPILED"]:
        cls_model = torch.compile(cls_model)
    if config["MODEL"]["EMA"]["COMPILED"] and ema_cls_model is not None:
        ema_cls_model = torch.compile(ema_cls_model)

    return cls_model, ema_cls_model


def define_loss(config: Any, device: torch.device) -> nn.CrossEntropyLoss:
    if config["TRAIN"]["LOSSES"]["NAME"] == "CrossEntropyLoss":
        if config["TRAIN"]["LOSSES"]["LABEL_SMOOTHING_FACTOR"] != 0.0:
            criterion = nn.CrossEntropyLoss(label_smoothing=config["TRAIN"]["LOSSES"]["LABEL_SMOOTHING_FACTOR"])
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['NAME']} is not implemented.")
    criterion = criterion.to(device)

    return criterion


def define_optimizer(config: Any, model: nn.Module) -> optim.SGD:
    if config["TRAIN"]["OPTIM"]["NAME"] == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              config["TRAIN"]["OPTIM"]["LR"],
                              config["TRAIN"]["OPTIM"]["MOMENTUM"],
                              config["TRAIN"]["OPTIM"]["DAMPENING"],
                              config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"],
                              config["TRAIN"]["OPTIM"]["NESTEROV"])
    else:
        raise NotImplementedError(f"Optimizer {config['TRAIN']['OPTIM']['NAME']} is not implemented.")

    return optimizer


def define_scheduler(config: Any, optimizer: optim.SGD) -> lr_scheduler.CosineAnnealingLR:
    if config["TRAIN"]["LR_SCHEDULER"]["NAME"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   config["TRAIN"]["LR_SCHEDULER"]["T_MAX"],
                                                   config["TRAIN"]["LR_SCHEDULER"]["ETA_MIN"],
                                                   config["TRAIN"]["LR_SCHEDULER"]["LAST_EPOCH"])
    else:
        raise NotImplementedError(f"Scheduler {config['TRAIN']['LR_SCHEDULER']['NAME']} is not implemented.")

    return scheduler


def train(
        cls_model: nn.Module,
        ema_cls_model: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.SGD,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_data_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":6.6f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses, acc1, acc5],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    cls_model.train()

    # Define loss function weights
    loss_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["WEIGHT"]).to(device)

    # Initialize data batches
    batch_index = 0
    # Set the dataset iterator pointer to 0
    train_data_prefetcher.reset()
    # Record the start time of training a batch
    end = time.time()
    # load the first batch of data
    batch_data = train_data_prefetcher.next()

    while batch_data is not None:
        # Load batches of data
        images = batch_data["images"].to(device, non_blocking=True)
        target = batch_data["target"].to(device, non_blocking=True)

        # Record the time to load a batch of data
        data_time.update(time.time() - end)

        # Initialize generator gradients
        cls_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            output = cls_model(images)
            loss = criterion(output, target)
            loss = torch.sum(torch.mul(loss_weight, loss))

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        if config["MODEL"]["EMA"]["ENABLE"]:
            # update exponentially averaged model weights
            ema_cls_model.update_parameters(cls_model)

        # measure accuracy and record loss
        # Get batch size
        batch_size = images.size(0)
        top1, top5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        acc1.update(top1[0].item(), batch_size)
        acc5.update(top5[0].item(), batch_size)

        # Record the total time of training a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output training log information once
        if batch_index % config["TRAIN"]["PRINT_FREQ"] == 0:
            # write training log
            iters = batch_index + epoch * batches
            writer.add_scalar("Train/Loss", loss.item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # Add 1 to the number of data batches
        batch_index += 1


if __name__ == "__main__":
    main()
