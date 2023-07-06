import time
from functools import partial
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .meter import AverageMeter, ProgressMeter
from .metric import calc_accuracy

__all__ = ["train", "evaluate"]

logger = getLogger(__name__)


def do_one_iteration(
    sample: Dict[str, Any],
    model: nn.Module,
    criterion: Any,
    device: str,
    iter_type: str,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:

    if iter_type not in ["train", "evaluate"]:
        message = "iter_type must be either 'train' or 'evaluate'."
        logger.error(message)
        raise ValueError(message)

    if iter_type == "train" and optimizer is None:
        message = "optimizer must be set during training."
        logger.error(message)
        raise ValueError(message)

    inputs = sample["images"].to(device)
    t = sample["labels"]

    batch_size = inputs.shape[0]
    seq_length = inputs.shape[1]

    t = t.view(batch_size * seq_length, -1)
    # t = t.to(device)
    gt = t.max(dim=1)[1]
    gt = gt.to(device)
    # gt = t.max(dim=1)
    # gt = [np.where(gt[0] == event)[0][0] if len(np.where(gt[0] == event)[0]) > 0 else -1 for event in range(1, 9)]
    # gt = np.array(gt)
    # gt = torch.from_numpy(gt)
    # gt = gt.to(device)

    # compute output and loss
    if iter_type == "train":
        output = model(inputs)
    else:
        if seq_length != 64:
            output = []
            for i in range(0, seq_length, 64):
                if i + 64 > seq_length:
                    supplement = torch.zeros(batch_size, i + 64 - seq_length, 3, 160, 160).to(device)
                    input = torch.cat([inputs[:, i:], supplement], dim=1)
                    output.append(model(input))
                else:
                    output.append(model(inputs[:, i : i + 64]))
            output = torch.cat(output, dim=1)
            output = output[:, :seq_length, :]
            output = output.view(batch_size * seq_length, -1)
        else:
            output = model(inputs)
    # output = model(inputs)

    loss = 0.0
    accs = 0.0

    if isinstance(output, list):
        for o in output:
            o = o.view(batch_size * seq_length, -1)
            loss += criterion(o, gt)
        gt = gt.to(device)
        accs = calc_accuracy(output[-1], gt, topk=(1,))
        loss = torch.sum(loss)
    else:
        output = output.to(device)
        loss = criterion(output, gt)
        accs = calc_accuracy(output, gt, topk=(1,))
        loss = torch.sum(loss)

    # measure accuracy and record loss
    acc1 = accs[0]

    if iter_type == "train" and optimizer is not None:
        # compute gradient and do SGD step
        gt = gt.cpu().numpy()

        if isinstance(output, list):
            pred = output[-1].max(dim=1)[1].cpu().numpy()
        else:
            pred = output.max(dim=1)[1].cpu().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if iter_type == "evaluate":
        # gt = t.argmax(dim=0)[0:-1].cpu().numpy()[1::2]
        # pred = output.argmax(dim=0)[0:-1].cpu().numpy()[1::2]
        # gt = t.argmax(dim=0)[0:-1].cpu().numpy()
        # pred = output.argmax(dim=0)[0:-1].cpu().numpy()
        gt = t.argmax(dim=1).cpu().numpy()
        gt = [np.where(gt == event)[0][0] if len(np.where(gt == event)[0]) > 0 else -1 for event in range(1, 9)]
        preds = output.argmax(dim=1).cpu().numpy()
        pred = [np.where(preds == event)[0][0] if len(np.where(preds == event)[0]) > 0 else -1 for event in range(1, 9)]
        gt = np.array(gt)
        pred = np.array(pred)
        another_pred = []
        for i in range(1, 9):
            for search_len in reversed(range(1, 10)):
                search_list = [i for j in range(search_len)] + [(i + 1) for j in range(search_len)]
                for j in range(len(preds) - search_len):
                    if np.all(preds[j : j + search_len] == search_list):
                        another_pred.append(j + search_len)
                        break
                if len(another_pred) > i:
                    break
            if len(another_pred) <= i:
                another_pred.append(np.where(preds == i)[0][0] if len(np.where(preds == i)[0]) > 0 else -1)
        another_pred = np.array(another_pred)
        pred = another_pred
        print(preds, gt)
    return batch_size, loss.item(), acc1, gt, pred


def train(
    loader: DataLoader,
    model: nn.Module,
    criterion: Any,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
) -> Tuple[float, float, float]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size, loss, acc1, gt, pred = do_one_iteration(sample, model, criterion, device, "train", optimizer)

        losses.update(loss, batch_size)
        top1.update(acc1, batch_size)

        # save the ground truths and predictions in lists
        gts += list(gt)
        preds += list(pred)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)

    progress.display(i)

    # calculate F1 Score
    f1s = f1_score(gts, preds, average="macro")

    return losses.get_average(), top1.get_average(), f1s


def evaluate(
    loader: DataLoader, model: nn.Module, criterion: Any, device: str
) -> Tuple[float, float, float, List[int]]:
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []
    corrects = []

    # calculate confusion matrix
    n_classes = loader.dataset.get_n_classes()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in loader:
            batch_size, loss, acc1, gt, pred = do_one_iteration(sample, model, criterion, device, "evaluate")

            correct = correct_preds(pred, gt)

            losses.update(loss, batch_size)
            top1.update(acc1, batch_size)

            # keep predicted results and gts for calculate F1 Score
            gts += list(gt)
            preds += list(pred)
            corrects.append(correct)

    f1s = f1_score(gts, preds, average="macro")
    PCE = np.round(np.mean(corrects), decimals=3) * 100
    acc_per_phase = np.round(np.mean(corrects, axis=0), decimals=3) * 100

    return losses.get_average(), top1.get_average(), f1s, PCE, acc_per_phase


def correct_preds(pred_events, gt_events, tol=-1):
    """
    Gets correct events in full-length sequence using tolerance based on number of frames from address to impact.
    Used during validation only.
    :param probs: (sequence_length, 9)
    :param labels: (sequence_length,)
    :return: array indicating correct events in predicted sequence (8,)
    """
    if tol == -1:
        tol = int(max(np.round((gt_events[5] - gt_events[0]) / 30), 1))
    deltas = np.abs(gt_events - pred_events)
    correct = (deltas <= tol).astype(np.uint8)
    return correct
