import argparse
import datetime
import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam  # Optimizer for doTraining
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from util.util import enumerateWithEstimate  # Runtime estimation(?)
# from dsets import LunaDataset
# from model import LunaModel  # The old model

from my_model import UNetWrapper, SegmentationAugmentation
from my_dsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt

from util.logconf import logging

import shutil  # For saving the best model
import hashlib  # For adding unique SHA-1 hex code to saved models in case file name is renamed wrong

log = logging.getLogger(__name__)   # Not sure. To make custom log?
log.setLevel(logging.INFO)          #
log.setLevel(logging.DEBUG)         #

# LunaTrainingApp metrics index values
    # Indexes for label, prediction, loss, size saving for each data sample.
    # Used to find loss of batch (computerBatchLoss) and logMetrics to save in metrics_t / metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2

# SegmentationTrainingApp metrics index values
METRICS_LOSS_NDX = 3
METRICS_TP_NDX = 4
METRICS_FN_NDX = 5
METRICS_FP_NDX = 6

METRICS_SIZE = 7

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]  # If not provided with args, take from command line

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processed for background data loading',
                            default=4,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )

        parser.add_argument('--balanced',
                            help="Balance the training data to half positive, half negative",
                            action='store_true',
                            default=False,
                            )

        parser.add_argument('--augmented',
                            help="Apply all augmentations at once",
                            default=False,
                            type=bool,
                            )
        parser.add_argument('--augment_flip',
                            help="Apply flip augmentation",
                            default=False,
                            type=bool,
                            )
        parser.add_argument('--augment_offset',
                            help="Apply offset augmentation",
                            default=False,
                            type=bool,
                            )
        parser.add_argument('--augment_scale',
                            help="Apply scale augmentation",
                            default=False,
                            type=bool,
                            )
        parser.add_argument('--augment_rotate',
                            help="Apply rotation augmentation",
                            default=False,
                            type=bool,
                            )
        parser.add_argument('--augment_noise',
                            help="Apply noise augmentation",
                            default=False,
                            type=bool,
                            )

        parser.add_argument('--tb-prefix',
                            default='diffaugments',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )

        parser.add_argument('--comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dlwpt',
                            )

        self.cli_args = parser.parse_args(sys_argv)  # The args from command line interface (CLI) to pass to
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')  # Start time

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()                       # Setting device to GPU
        self.device = torch.device("cuda" if self.use_cuda else "cpu")  #

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()  # init after model so optimizer knows to look in GPU for parameters

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True  # 0.5 probability in dsets
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

    def initModel(self):
        model = LunaModel()  # Using LunaModel() CNN from model.py
        if self.use_cuda:
            log.info("Using CUDA; {} devices".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)  # For running model on multiple GPUs
            model = model.to(self.device)  # Send model parameters to GPU(s)
        return model


    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)  # Sensible lr and momentum


    def initTrainD1(self):
        train_ds = LunaDataset(  # Bring in LunaDataset, containing CT scans in neat tensor form
            val_stride=10,
            isValSet_bool=False,
            ratio_int=int(self.cli_args.balanced)
        )

        batch_size = self.cli_args.batch_size  # If --batch-size= specified in the input in bash
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()  # To account for parallel GPUs

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,  # pin_memory ensures memory is kept on GPU(s), reducing slow-downs
        )

        return train_dl

    def initValD1(self):
        val_ds = LunaDataset(  # Bring in LunaDataset, containing CT scans in neat tensor form
            val_stride=10,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.batch_size  # From CLI input
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()  # Parallel GPUs allow larger batch sizes

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda  # pin_memory ensures memory is kept on GPU(s), reducing slow-downs
        )

        return val_dl

    def initTensorboardWriters(self):  # For tensorboard stuff
        if self.trn_writer is None:
            print("self.cli_args.comment: ", self.cli_args.comment)
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainD1()
        val_dl = self.initValD1()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()  # Specify that model is going to train not eval (so batch-norm, dropout, etc. knows)

        trnMetrics_g = torch.zeros(  # Initialises empty metrics array, records per-sample bhvr, sends to logMetric
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(  # Collect batch data, and estimation for how long it will take
            train_dl, "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:  # For each batch...
            self.optimizer.zero_grad()  # Free grads for next batch

            loss_var = self.computeBatchLoss(  # Compute loss for batch
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()  # Back-propagate loss function to get gradients
            self.optimizer.step()  # Update weights and biases

        self.totalTrainingSamples_count += len(train_dl.dataset)  # Record total samples trained on thus far

        return trnMetrics_g.to('cpu')  # Return metrics from GPU to CPU for printing/logging

    def doValidation(self, epoch_ndx, val_dl):  # Same as doTraining, minus back-propagation and updating parameters
        with torch.no_grad():  # Ensures .backward() does not use validation branch
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl, "E{} Validation".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )

            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)  # Log info to valMetrics_g

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        """
        For Classification model
        Computes loss over batch,
        and saves per-sample info about model outputs e.g. % correct outputs per class, into metrics_g*
        *Allows recording of important metrics when troubleshooting model issues
        :param batch_ndx:
        :param batch_tup:
        :param batch_size:
        :param metrics_g:
        :return:
        """

        input_t, label_t, _series_list, _center_list = batch_tup  # input, label output, series_uid, center_irc

        input_g = input_t.to(self.device, non_blocking=True)  # Send input/label data to GPU for faster loss computation
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)  # Calculated on GPU

        loss_func = nn.CrossEntropyLoss(reduction='none')  # reduction='none' gives the loss per sample
        loss_g = loss_func(logits_g, label_g[:, 1])  # [:, 1] gives list of 'it IS a nodule' class for batch

        start_mdx = batch_ndx * batch_size  # start index is total samples so far = batch size * number of batches
        end_ndx = start_mdx + label_t.size(0)  # since data has length label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_mdx:end_ndx] = label_g[:, 1].detach()  # Save true 'is nodule' labels per smp
        metrics_g[METRICS_PRED_NDX, start_mdx:end_ndx] = probability_g[:, 1].detach()  # Save model's prob of 'is nod'
        metrics_g[METRICS_LOSS_NDX, start_mdx:end_ndx] = loss_g.detach()  # Save model's losses per sample

        return loss_g.mean()  # Return single-value mean of loss function for batch


    def logMetrics(
            self,
            epoch_ndx,  # Used for printing logs
            mode_str,  # Specifies if metrics being logged are from training or validation
            metrics_t,  # Metrics info used for logging
            classificationThreshold=0.5
    ):
        self.initTensorboardWriters()  # Initialise Tensorboard
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold  # = 1 when non-nod (0 nod) label
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold  # = 1 when non-nod, (0 nod) model pred.

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())  # Number of true non-nodules
        pos_count = int(posLabel_mask.sum())  # Number of true nodules

        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())  # Num of non-nods correctly predicted
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())  # Num of nodules correctly predicted

        falseNeg_count = neg_count - trueNeg_count
        falsePos_count = pos_count - truePos_count

        precision = truePos_count / np.float32(truePos_count + falsePos_count)  # Fraction of pos flags being true
        recall = truePos_count / np.float32(truePos_count + falseNeg_count)  # Fraction of positives being detected

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()  # avg loss over ALL samples recorded
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()  # avg loss over non-nodules
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()  # avg loss over nodules

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100  # as %
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        metrics_dict['pr/precision'] = precision
        metrics_dict['pr/recall'] = recall
        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)  # F1 score

        log.info(
            ("E{} {:8} "
             + "{loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             + "{pr/precision:.4f} precision, "
             + "{pr/recall:.4f} recall, "
             + "{pr/f1_score:.4f} F1 score"
             ).format(epoch_ndx, mode_str, **metrics_dict,)
        )

        log.info(
            "E{} {:8f} {loss/neg:.4f} loss, {correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            .format(epoch_ndx, mode_str + '_neg', neg_correct=neg_correct, neg_count=neg_count, **metrics_dict,)
        )

        log.info(
            "E{} {:8f} {loss/pos:.4f} loss, {correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            .format(epoch_ndx, mode_str + '_pos', pos_correct=pos_correct, pos_count=pos_count, **metrics_dict,)
        )

        # TENSORBOARD STUFF:
        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x / 50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]  # If not provided with args, take from command line

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processed for background data loading',
                            default=4,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )

        parser.add_argument('--balanced',
                            help="Balance the training data to half positive, half negative",
                            action='store_true',
                            default=False,
                            )

        parser.add_argument('--augmented',
                            help="Apply all augmentations at once",
                            default=False,
                            type=bool,
                            )
        parser.add_argument('--augment_flip',
                            help="Apply flip augmentation",
                            default=False,
                            type=bool,
                            )
        parser.add_argument('--augment_offset',
                            help="Apply offset augmentation",
                            default=False,
                            type=bool,
                            )
        parser.add_argument('--augment_scale',
                            help="Apply scale augmentation",
                            default=False,
                            type=bool,
                            )
        parser.add_argument('--augment_rotate',
                            help="Apply rotation augmentation",
                            default=False,
                            type=bool,
                            )
        parser.add_argument('--augment_noise',
                            help="Apply noise augmentation",
                            default=False,
                            type=bool,
                            )

        parser.add_argument('--tb-prefix',
                            default='diffaugments',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )

        parser.add_argument('--comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='dlwpt',
                            )

        self.cli_args = parser.parse_args(sys_argv)  # The args from command line interface (CLI) to pass to
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')  # Start time

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True  # 0.5 probability in dsets
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()  # Setting device to GPU
        self.device = torch.device("cuda" if self.use_cuda else "cpu")  #

        self.segmentation_model, self.augmentation_model = self.initModel()
        self.optimizer = self.initOptimizer()  # init after model so optimizer knows to look in GPU for parameters

    def initModel(self):
        """
        The new model, incorporating segmentation and index layers in separate channels
        :return:
        """
        segmentation_model = UNetWrapper(
            in_channels=7,  # Number of channels is number of slices, 3 on either side ==> 3 + 1 + 3 = 7
            n_classes=1,  # Number of output channels
            depth=3,  # Network depth (number of down-samples)
            wf=4,  # Where number of filters in first layer is 2**wf, which doubles with each down-sampling
            padding=True,  # Apply padding so input shape = output shape when pooling
            batch_norm=True,  # Batch-normalise after activation functions
            up_mode='upconv',  # Transposed convolutions for up-sampling
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            segmentation_model = segmentation_model.to(self.device)  # Send model parameters to GPU(s)
            augmentation_model = augmentation_model.to(self.device)
            if torch.cuda.device_count() > 1:  # For running model on multiple GPUs
                segmentation_model = nn.DataParallel(segmentation_model, device_ids=range(torch.cuda.device_count()))
                augmentation_model = nn.DataParallel(augmentation_model, device_ids=range(torch.cuda.device_count()))

        return segmentation_model, augmentation_model

    def initOptimizer(self):
        return Adam(self.segmentation_model.parameters())  # Generally good optimiser, no need to fine-tune hyper-params

    def initTrainD1(self):
        train_ds = TrainingLuna2dSegmentationDataset(  # Bring in LunaDataset, containing CT scans in neat tensor form
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size  # If --batch-size= specified in the input in bash
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()  # To account for parallel GPUs

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,  # pin_memory ensures memory is kept on GPU(s), reducing slow-downs
        )

        return train_dl

    def initValD1(self):
        val_ds = Luna2dSegmentationDataset(  # Bring in Luna2dSegmentationDataset, containing CT scans in neat tensor form
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size  # From CLI input
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()  # Parallel GPUs allow larger batch sizes

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda  # pin_memory ensures memory is kept on GPU(s), reducing slow-downs
        )

        return val_dl

    def initTensorboardWriters(self):  # For tensorboard stuff
        if self.trn_writer is None:
            # print("self.cli_args.comment: ", self.cli_args.comment)
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_seg-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_seg-' + self.cli_args.comment)

    def main(self):

        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainD1()
        val_dl = self.initValD1()

        self.validation_cadence = 5
        best_score = 0.0
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            # print("Size of trnMetrics_g: ", np.shape(self.doTraining(epoch_ndx, train_dl)))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:

                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)  # Score = recall
                best_score = max(best_score, score)  # Record the best score so far

                self.saveModel('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(  # Initialises empty metrics array, records per-sample bhvr, sends to logMetric
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        self.segmentation_model.train()
        # Specify that model is going to train not eval (so batch-norm, dropout, etc. knows)
        train_dl.dataset.shuffleSamples()

        batch_iter = enumerateWithEstimate(  # Collect batch data, and estimation for how long it will take
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:  # For each batch...
            self.optimizer.zero_grad()  # Free grads for next batch

            loss_var = self.computeBatchLoss(  # Compute loss for batch
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()  # Back-propagate loss function to get gradients

            self.optimizer.step()  # Update weights and biases

        self.totalTrainingSamples_count += trnMetrics_g.size(1)  #  = len(train_dl.dataset)
        # Records total samples trained on thus far

        return trnMetrics_g.to('cpu')  # Return metrics from GPU to CPU for printing/logging

    def doValidation(self, epoch_ndx, val_dl):  # Same as doTraining, minus back-propagation and updating parameters
        with torch.no_grad():  # Ensures .backward() does not use validation branch
            self.segmentation_model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )

            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)  # Log info to valMetrics_g

        return valMetrics_g.to('cpu')

    def diceloss(self, prediction_g, label_g, epsilon=1):
        """
        Computes the Dice loss of the predicted mask. Essentially the F1 Score, but where the population is the pixels
        in the image, rather than the CT chunks from the classification model.
        Since the population is contained within a single training sample, the dice loss can be used for training,
        (F1 score couldn't as it was only calculable when assessing how well the model performed during validation).
        Dice ratio = 2 * true positive pixels / (number of predicted positive pixels + number of actual positive pixels)
        Dice loss = 1 - Dice ratio (high overlap ~0, low overlap ~1)
        :param prediction_g: Predicted *non-Boolean* mask of nodules and not nodules (Boolean is indifferentiable)
        :param label_g: The actual Boolean mask of nodule and not nodules
        :param epsilon: Prevent dividing by zero error in case diceLabel and/or dicePrediction don't exist
        :return: Dice loss
        """
        diceLabel_g = label_g.sum(dim=[1, 2, 3])  # Sums all the positively-labelled points, per batch item.
        dicePrediction_g = prediction_g.sum(dim=[1, 2, 3])  # Sums all the positively-predicted points, per batch item
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1, 2, 3])  # sums all the correctly positively predicted points, per batch item

        diceRatio_g = (2 * diceCorrect_g + epsilon) / (diceLabel_g + dicePrediction_g + epsilon)

        return 1 - diceRatio_g

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g, classificationThreshold=0.5):
        """
        Compute loss function for a batch, for the new segmentation model
        :param batch_ndx:
        :param batch_tup:
        :param batch_size:
        :param metrics_g: Stores the dice loss, true positive, false negative and false positive. For logging/plotting
        :param classificationThreshold:
        :return: diceLoss + fnLoss * 8
        Because getting all the positives pixels correct (high recall) is more important than getting all the negative
        pixels correct (to minimise false negative diagnoses), a weighted sum is used.
            diceLoss is the loss for overall prediction accuracy, ensuring false positives aren't too common
            fnLoss is the loss for true positive prediction accuracy.
                It is multiplied by 8 to ensure the model is minimising false negatives, which is the most important
                accuracy measurement.
                Positive pixels have more influence when back-propagating to minimise loss.
            Note: This works with Adam optimiser since it can change the learning rate accordingly which prevents the
            true positive weight to overpower the overall accuracy, but doing this with SGD would likely result in every
            pixel coming back positive.
        """
        input_t, label_t, series_list, _slice_ndx_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)  # Transfer input and labels to GPU
        label_g = label_t.to(self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.segmentation_model(input_g)

        diceLoss_g = self.diceloss(prediction_g, label_g)
        fnLoss_g = self.diceloss(prediction_g * label_g, label_g)  # Dice loss of true predicted positives and actual positives

        start_ndx = batch_ndx * batch_size  # Start and end index in metrics_g array to record data
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1] > classificationThreshold).to(torch.float32)
            # Convert to Boolean rather than [0.,1.] float, then convert to float32 for multiplication

            tp = (      predictionBool_g    *   label_g).sum(dim=[1, 2, 3])   # true positives
            fn = ((1 -  predictionBool_g)   *   label_g).sum(dim=[1, 2, 3])   # false negatives
            fp = (      predictionBool_g    * (~label_g)).sum(dim=[1, 2, 3])  # false positives (~ means complement)

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return diceLoss_g.mean() + fnLoss_g.mean() * 8

    def logImages(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval()

        images = sorted(dl.dataset.series_list)[:12]
        # Takes the first 12 CT's from series_list (sorted ensures they are always the same 12)

        for series_ndx, series_uid in enumerate(images):
            ct = getCt(series_uid)

            for slice_ndx in range(6):
                ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5  # Selects indexes 0, N//5 2N//5, 3N//5, 4N//5, N
                sample_tup = dl.dataset.getitem_fullSlice(series_uid, ct_ndx)  # Fetch the slice data of index

                ct_t, label_t, series_uid, ct_ndx = sample_tup  # Unpack tuple data into separate variables

                input_g = ct_t.to(self.device).unsqueeze(0) # Send ct_t to gpu (faster to run model on gpu?)
                label_g = label_t.to(self.device).unsqueeze(0)  # Send label_t to gpu

                prediction_g = self.segmentation_model(input_g)[0]  # prediction = output of model from input
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                # Send to cpu from gpu.
                # .detach(), detaches it from the computational graph pytorch makes for calculating gradients.
                # This is done because no gradient will be back-propagated through this variable.
                # calling .detach() tells this to pytorch.
                # Doing this uses less memory
                # Convert torch tensor to numpy and take [0] to remove extra brackets
                # > 0.5 to convert to Boolean type

                label_a = label_g.cpu().numpy()[0][0] > 0.5
                # Send to cpu from gpu, if not on cpu memory already
                # Convert torch tensor to numpy, remove extra brackets, then convert to Boolean

                ct_t[:-1, :, :] /= 2000  # Divide all values by 2000, so range [-1000,1000] --> [-0.5, 0.5]
                ct_t[:-1, :, :] += 0.5  # Values in range [0, 1]

                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()  # Set to first 7 slices of ct_t
                # shape = (512, 512, 7) (assuming 3 context slices per side)

                image_a = np.zeros((512, 512, 3), dtype=np.float32)  # Make empty image
                image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))  # Set to ctSlice_a and matrix of tuples
                # image_a currently has values in [0, 1]

                # Red channel
                image_a[:, :, 0] += prediction_a & (1 - label_a)  # False positive
                image_a[:, :, 0] += (1 - prediction_a) & label_a  # False negative
                # Adds values either {0, 1} on top of image_a initially with values [0, 1]

                # Green channel
                image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5  # False negative has 0.5 (either {0, 0.5})
                image_a[:, :, 1] += prediction_a & label_a  # True positive has 1.0 (either {0, 1.0})
                # Adds values either {0, 0.5, 1} on top of image_a initially with values [0, 1]

                # For false positives, R = 1, G = 0.5 (Red colour)
                # False negatives      R = 1, G = 0   (Orange colour)
                # True positives       R = 0, G = 1.0 (Green colour)

                # Image has RGB channels with values between [0, 2]
                image_a *= 0.5
                image_a.clip(0, 1, image_a)  # Clamped to (0,1) in case augmentation pushes values out of range
                # Image now has RBG channels with values between [0, 1]
                # Final result: CT image at half-intensity, overlaid with red, orange and green pixels.

                writer = getattr(self, mode_str + '_writer')  # Add image_a to Tensorboard
                writer.add_image(
                    f'{mode_str}/{series_ndx}_prediction{slice_ndx}',
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC'  # Order of axes is Height, Width, Channel
                )

                if epoch_ndx == 1:  # Label image never changes, so only add image in first epoch.

                    label_image_a = np.zeros((512, 512, 3), dtype=np.float32)  # Make empty image
                    label_image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))  # Set to ctSlice_a and matrix of tuples
                    # label_image_a currently has values in [0, 1]

                    # Green channel
                    label_image_a[:, :, 1] += label_a  # True positive has 1.0 (either {0, 1.0})
                    # Adds 1 on top of label_image_a (initially with values [0, 1]) at nodule pixels

                    # Label image has RGB channels with values between [0, 2]
                    label_image_a *= 0.5
                    label_image_a.clip(0, 1, label_image_a)
                    # Clamped to (0,1) in case augmentation pushes values out of range
                    # Label image now has RBG channels with values between [0, 1]
                    # Final result: CT image at half-intensity, overlaid with green pixels at the actual positive nodules.

                    writer.add_image(  # Add label_image_a to Tensorboard
                        f'{mode_str}/{series_ndx}_label_{slice_ndx}',
                        label_image_a,
                        self.totalTrainingSamples_count,
                        dataformats='HWC'  # Order of axes is Height, Width, Channel
                    )

                writer.flush()  # Helps prevent Tensorboard not get confused

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        """
        Ran every validation test. Creates a temporary metrics_dict dictionary, calculates the performance metrics for
        at the current epoch_ndx and logs it all.
        :param epoch_ndx:
        :param mode_str:
        :param metrics_t:
        :return: The score, which is the recall since the segmentation model needs to be able to recall every positive
        nodule.
        """
        log.info("E{} {}".format(epoch_ndx, type(self).__name__))  # ?

        metrics_a = metrics_t.detach().numpy()  # Detach from pytorch computational graph

        metrics_dict = {}

        sum_a = metrics_a.sum(axis=1)  # Sum total of the number of true positives, false negatives, etc.
        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]  # Total number of actual nodule pixels
        metrics_dict['percent_all/tp'] = sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100  # Prevent div by zero err
        metrics_dict['percent_all/fn'] = sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100
        # percent_all/fp may be larger than 100% because there can be more false positives than actual nodule pixels
        # Regardless, this is an informative metric since it gives how many false positives there are, relative
        # to actual nodules.

        precision   = metrics_dict['pr/precision'] \
                    = sum_a[METRICS_TP_NDX] / (sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX] or 1)

        recall      = metrics_dict['pr/recall'] \
                    = sum_a[METRICS_TP_NDX] / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / ((precision + recall) or 1)

        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()  # Mean loss

        log.info(("E{} {:8} "  # Log/save the information
                  + "{loss/all:.4f} loss, "
                  + "{pr/precision:.4f} precision, "
                  + "{pr/recall:.4f} recall, "
                  + "{pr/f1_score:.4f} f1 score"
                  ).format(epoch_ndx, mode_str, **metrics_dict))

        log.info(("E{} {:8} "  # Log/save the true pos, false neg and false pos
                  + "{loss/all:.4f} loss,"
                  + "{percent_all/tp:-5.1f}% true pos, {percent_all/fn:-5.1f}% false neg, {percent_all/fp:-9.1f}% false pos"
                  ).format(epoch_ndx, mode_str + '_all', **metrics_dict))

        # Tensorboard stuff
        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'  # Segmentation model

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/recall']

        return score

    def initTensorboardWriters(self):  # For tensorboard stuff
        if self.trn_writer is None:
            # print("self.cli_args.comment: ", self.cli_args.comment)
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '-trn_seg-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_seg-' + self.cli_args.comment)

    def main(self):

        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainD1()
        val_dl = self.initValD1()

        self.validation_cadence = 5
        best_score = 0.0
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            # print("Size of trnMetrics_g: ", np.shape(self.doTraining(epoch_ndx, train_dl)))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:

                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)  # Score = recall
                best_score = max(best_score, score)  # Record the best score so far

                self.saveModel('seg', epoch_ndx, score == best_score)

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        file_path = os.path.join('data-unversioned', 'part2', 'models',
                                 '{}_{}_{}.{}.state'.format(
                                     type_str,  # Type of model. In this case, segmentation
                                     self.time_str,  # Current time
                                     self.cli_args.comment,  # Any arguments from the command line when model was ran
                                     self.totalTrainingSamples_count  # Number of samples trained on
                                 ))

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),  # the weights and biases of the model
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),  # Optimizer hyperparameters like learning rate + momentum
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)
        log.info("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'data-unversioned', 'part2', 'models', self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')
            # type_str:                 Type of model. In this case, segmentation
            # self.time_str:            Current time
            # self.cli_args.comment:    The comment from command line when training started

            shutil.copyfile(file_path, best_path)  # Copies the model saved at file_path, saves copy in best_path

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA-1: " + hashlib.sha1(f.read()).hexdigest())
            # Logs a unique SHA-1 cyptography code. In case we need to debug exactly which model we're working with,
            # For instance, if a file is renamed incorrectly, this is an identifiable code

if __name__ == '__main__':  # Makes instance of application and starts main thread
    # LunaTrainingApp().main()
    SegmentationTrainingApp().main()

