import functools
import os.path
from collections import namedtuple
import csv
import glob
import os
import random
import math

import SimpleITK as sitk  # Convert MetalIO data format of CT scans to numpy arrays
import numpy as np

import torch
import torch.cuda
import torch.nn.functional as F  # (for Luna dataset)
from torch.utils.data import Dataset

import copy  # (for Luna dataset) copying candidateInfo_list from cached data

from util.disk import getCache
from util.logconf import logging
from util.util import XyzTuple, xyz2irc


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2')

MaskTuple = namedtuple('MaskTuple',
                       'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')

CandidateInfoTuple = namedtuple(  # Used to create tuples to use as an interface for the luna
    'CandidateInfoTuple',
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')


@functools.lru_cache(1)  # Load data in RAM
def getCandidateInfoList(requireOnDisk_bool=True):  # Returns clean dictionary data, accounting for coord error
    """
    Collects .mhd data files for all nodule candidates (so including non-nodules) that are downloaded in /luna
    First opens annotations_with_malignancy.csv, like annotations.csv but only contains annotations of ACTUAL nodules
        but additionally contains information on whether the nodule is malignant or benign
        (also filtered out any duplicate annotation of same nodule but labelled in multiple index layers).
        For each label,
            Appends all the info to candidateInfo_list as CandidateInfoTuple type
            Saves diameter info to diameter_dict (used for checking error in second part)
    Second, opens the candidates.csv file, containing the list of all candidates, including ACTUAL nodules and filters
        out the actual nodules which have already been appended (plus addressing possible errors which may arise)
        candidates.csv (should) contains all the nodules in annotations_with_malignancy.csv, within an error of human
        labelling.
        For each candidate,
            Checks that delta between the files' coordinates are within an acceptable error. If within error, assumes
            they are describing the same nodule.
            Otherwise, is ignored since it would mean candidates do not contain a label from annotations_with_malignancy
            - a rare case but must be addressed to ensure no dubious data.
            Appends only the non-nodule candidates to candidateInfo_list as CandidateInfoTuple type, since actual
            nodules are already added in first part.
    requireOnDisk_bool=True implies only collects .mhd for data downloaded
    """
    mhd_list = glob.glob('E:/Datasets/deep-learning-with-pytorch/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}  # dictionary of nodule candidate data available

    # diameter_dict = {}  # dictionary of ALL nodule data in annotations.csv file
    candidateInfo_list = []

    with open('E:/Datasets/deep-learning-with-pytorch/luna/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:  # First line headings. row = (uid, x, y, z, diam, isMal_str)
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:  # If uid data not on disk then do nothing
                continue

            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])  # =(x, y, z)
            annotationDiameter_mm = float(row[4])
            isMal_bool = {'False': False, 'True': True}[row[5]]  # Converts isMal_str in data to boolean

            candidateInfo_list.append(  # Add to candidateInfo_list
                CandidateInfoTuple(
                    True,  # isNodule_bool
                    True,  # hasAnnotation_bool
                    isMal_bool,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )

            # diameter_dict.setdefault(series_uid, []).append(  # = (uid, (x, y, z), diameter)
            #     (annotationCenter_xyz, annotationDiameter_mm)
            # )

    # Appending non-nodules to candidateInfo_list and checks for error between x, y, z values
    with open('E:/Datasets/deep-learning-with-pytorch/luna/candidates.csv', "r") as f:  # Append rest of (non-nodule) data from candidates.csv
        # print(presentOnDisk_set)
        for row in list(csv.reader(f))[1:]:  # row = (uid, x, y, z, [1(malignant) or 0(benign)])
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:  # If uid data not on disk then do nothing
                # if series_uid == '1.3.6.1.4.1.14519.5.2.1.6279.6001.100398138793540579077826395208':
                #     print("Not being added")
                #     print(series_uid)
                continue

            isNodule_bool = bool(int(row[4]))  # True (is nodule) or False (not nodule)
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])  # = (x, y, z)

            # candidateDiameter_mm = 0.0
            # for annotation_tup in diameter_dict.get(series_uid, []):  # annotation_tup = ((x, y, z), diameter)
            #     annotationCenter_xyz, annotationDiameter_mm = annotation_tup
            #     for i in range(3):
            #         delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
            #         if delta_mm > annotationDiameter_mm / 4:  # if err btwn csv files larger than accepted
            #             break  # candidate_Diameter = 0.0 so not counted as nodule. Should be rare
            #     else:
            #         candidateDiameter_mm = annotationDiameter_mm  # Make equal
            #         break

            if not isNodule_bool:
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False,  # isNodule_bool
                        False,  # hasAnnotation_bool (not in annotations_with_malignancy)
                        False,  # isMal_bool
                        0.0,  # candidateDiameter_mm = 0
                        series_uid,
                        candidateCenter_xyz,
                    )
                )

    candidateInfo_list.sort(reverse=True)  # Sorted from the largest diameter to the smallest diameter
                                               # non-nodule samples, which have 0.0 diameter are last
    return candidateInfo_list


@functools.lru_cache(1)  # Prevents initialising Ct from causing bottleneck
def getCandidateInfoDict(requireOnDisk_bool=True):
    """
    Sorts the candidateInfo_list data into dictionary form, with the series_uid as the key, to make fetching data of candidates
    from specific CT easier.
    :param requireOnDisk_bool: Requires that the dictionary should only contain candidates present on the disk
    (in a specific directory)
    :return: candidateInfo_dict
    """
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid, []).append(candidateInfo_tup)
        # Appends candidateInfo_tup to its series_uid category in dictionary
        # If candidateInfo_tup.series_uid is not yet in the list, it's added to a fresh dictionary key

    return candidateInfo_dict

class Ct:  # Load CT scan data of given uid
    def __init__(self, series_uid):
        # print("glob.glob('luna/subset*/{}.mhd'.format(series_uid)): ", glob.glob('luna/subset*/{}.mhd'.format(series_uid)))
        mhd_path = glob.glob(
            'E:/Datasets/deep-learning-with-pytorch/luna/subset*/{}.mhd'
            .format(series_uid))[0]  #subset not specified, but will be unique

        ct_mhd = sitk.ReadImage(mhd_path)  # Reads data at mhd_path, and it will implicitly find the .raw file
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)  # (hu stand for HU units)
        # PyTorch likes float32
        # Convert CT image to array

        self.series_uid = series_uid

        # Fetch data for coordinate conversion from patient coords (xyz) to index row column (IRC) voxels
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)  # getDirection returns 9 element array

        # Making a self.positive_mask for a CT scan
        candidateInfo_list = getCandidateInfoDict()[self.series_uid]

        self.positiveInfo_list = [candidate_tup for candidate_tup in candidateInfo_list if candidate_tup.isNodule_bool]
        #   Make a list of the actual nodules in the nodule candidates list
        #   (filters candidateInfo_list to the actual nodules)

        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1, 2))
                                 .nonzero()[0].tolist())
        #   List of indexes (z axis) containing nodules (since many likely won't have any)
        #   sum(axis=(1,2)) takes sum of all row + column (indexes 1,2 of [I,R,C] values for each index
        #   .nonzero()      returns indexes of slices with nonzero sum (so contains nodule(s))
        #   [0]             taken since nonzero() returns a tuple like ([ ])
        #   .tolist()       convert from numpy type to list type


    def buildAnnotationMask(self, positiveInfo_list, threshold_hu=-700):
        """
        Build the Boolean-type mask for the CT, with all initially negative values.
        For each nodule in the CT:
            Translates xyz coordinate to irc.
            Constructs a cube, nodule_cube defined as a 3D array with initially all positive values around the nodule
            centre, which ends when the hu_a at a middle point of a side of the cube is less than a threshold value,
            threshold_hu.
            It then filters the cube array by the same threshold_hu, changing every element with a hu_a less than
            threshold_hu to negative.
            (Mask) ==> (Mask ∪ nodule_cube)
        :param positiveInfo_list: List of info for each nodule, including xyz coordinate
        :param threshold_hu: Threshold value defining when nodule becomes not nodule
        :return:
        """

        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool_)  # all-false boolean array with same size as CT scan

        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )

            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2  # Start with 2 on either side of centre
            try:  # Expand radius until HU value on one of the vertices < threshold, then stops
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:  # Ends when there is a 1-voxel thick border of low-density tissue (on side closest to centre)
                    index_radius += 1
            except IndexError:  # If index is outside of scope (size) of tensor, stop and reduce radius back
                index_radius -= 1
                # (note that if ci = 0 this MAY bring an error,
                # but there probably aren't any nodule centres in 0th index in our data)

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            boundingBox_a[
                ci - index_radius: ci + index_radius + 1,
                cr - row_radius: cr + row_radius + 1,
                cc - col_radius: cc + col_radius + 1] = True

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)
        # Boolean CT-scan-size array, where datapoints are
        # true if val at coord is within boundary box of nodule AND greater than min threshold
        # false everywhere else

        return mask_a

    def getRawCandidate(self, center_xyz, width_irc):
        """
        Extracts small chunk of data surrounding a central coordiante (of nodule) from CT
        :param center_xyz: Central coordinate (of nodule) in x,y,z spatial coordinates
        :param width_irc: Size of the chunk of CT data in i,r,c voxel coordinates
        :return: ct_chunk, pos_chunk, center_irc
        """
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)
        # Translate spatial coordinate to voxel coordinate for indexing

        slice_list = []   # Lists the list of indexes for each i,r,c axis
        for axis, center_val in enumerate(center_irc):  # For each coordinate axis, 0 (I), 1 (R), 2 (C)
            start_ndx = int(round(center_val - width_irc[axis]/2))  # min index in the chunk for the axis
            end_ndx = int(start_ndx + width_irc[axis])  # max index in chunk for the axis

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])
            # Asserts center_val is valid

            if start_ndx < 0:  # Address indexing error at lower boundary
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:  # Address indexing error at upper boundary
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))  # Values to slice from voxel CT data, across current axis

        ct_chunk = self.hu_a[tuple(slice_list)]  # Takes chunk of CT data
        pos_chunk = self.positive_mask[tuple(slice_list)]  # Takes corresponding chunk of mask (nodule or not nodule)

        return ct_chunk, pos_chunk, center_irc  # Returns CT and mask data, and the center coordinate in i,r,c


@functools.lru_cache(1, typed=True)  # Cached in MEMORY (keep ONE ct scan in memory)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    """
    Caches the ct_chunk from getRawCandidate for quick loading.
    Also cleans data.
    some CT scans have values less than air outside FOV of scan,
        set lower cap as air (-1000), upper cap as 1000 (incl. bones, implants, irrelevant)
        tumours are around ~0 HU (1g/cc)
    :param series_uid: The identifying CT UID
    :param center_xyz: Coordinate of chunk centre (i, r, c)
    :param width_irc: Size of chunk
    :return: ct_chunk, pos_chunk, center_irc from getRawCandidate
    """
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc


@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    """
    :param series_uid: the CT scan to find the size of
    :return: the size of the CT scan, and the z-axis (I) slices containing nodules
    """
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 contextSlices_count=3,
                 fullCt_bool=False,
                 ):
        """
        :param val_stride: How the training and validation sets are split. If validation set is needed
        (isValSet_bool=True), fetch every val_stride-th Ct scan.
        E.g. val_stride = 3 means fetch (0, 3, 6, ...)th series_uids.
        :param isValSet_bool: True if validation, false if training
        :param fullCT_bool: Chooses between two modes of validation. If true, every slice of each CT will be put into
        the dataset. If false, only the slices with nodules (contains positive_mask) are considered.
        First mode
            for testing end-to-end performance (detection when an entire CT scan is fed in), which is end goal
        Second mode
            for testing specifically the rates of true positives and false negatives of nodule detection
            (if model predicts the entire CT scan is false, it might be ~98% true, but it hasn't learnt anything about
            nodule detection which is the crucial role of the model)
            filtering to just the slices with nodules means we can get better statistics for true positives and false
            negatives
        """
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        if series_uid:
            self.series_list = [series_uid]  # For troubleshooting a specific CT scan
        else:
            self.series_list = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:  # If true, validation is considered, fetch validation set. If false, then fetch training set.
            assert val_stride > 0, val_stride  # Ensures val_stride is a valid value
            self.series_list = self.series_list[::val_stride]  # Sets the series_list to the validation set of series_uids
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]  # Sets the series_list to thr training set of series_uids
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = getCtSampleSize(series_uid)

            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in positive_indexes]

        self.candidateInfo_list = getCandidateInfoList()  # Fetches the list of nodule candidates (cached)

        series_set = set(self.series_list)
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set]
        # Filters out candidates from series that are not in the training/validation set

        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]
        # candidateInfo_list, but non-nodules filtered out, so only containing the actual nodules

        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.sample_list),
            len(self.pos_list),
        ))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        """
        :param ndx: Specifying the index of the sample in sample_list to return the CT scan and positive mask of.
        ndx is also wrapped in sample_list by % for when ndx is larger than the number of real data samples in
        sample_list. Augmentation will be applied which allows for expansion of the data set via small transformations
        applied to the real data.
        :return: getitem_fullslice(series_uid, slice_ndx) of the ndx-th sample
        """
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        """
        Get the full slice of the slice_ndx-th slice of CT, plus some additional CT slices before and after
        the slice_ndx-th slice. This gives spatial context across the index axis to the model.
        :param series_uid: Which CT to get.
        :param slice_ndx: Which slice from CT get
        :return: ct_t (a subset of the CT, centered at slice_ndx, with additional adjacent slices given for context),
         pos_t (the positive mask at slice_ndx). series_uid and slice_ndx also returned, for displaying/debugging.
         """
        ct = getCt(series_uid)
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1

        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            # Limits any of the indexes to 0 and the length of the CT scan
            # So any context_ndx where the index is outside the CT scan just takes the index of the first or last slice
            context_ndx = max(context_ndx, 0)  # if ndx < 0, set to 0
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)  # if ndx > max ndx = len-1, set to max ndx
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300000

    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self, candidateInfo_tup):
        """
        Get 64x64 section of the 96x96 portion of the candidate in CT given by candidateInfo_tup, plus some additional
        CT slices before and after to give spatial context across the index axis to the model.
        :param candidateInfo_tup: Information on the candidate such as series_uid, coordinate center_xyz, etc.
        :return: ct_t (a subset of the 96x96 chunk surrounding the candidate), pos_t (the positive mask of the 96x96
        chunk, series_uid and slice_ndx for displaying/debugging.
        """

        ct_a, pos_a, center_irc = getCtRawCandidate(  # 7x96x96 chunk around candidate centre
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (7, 96, 96),  # 3 additional slices on each side ==> 3 + 1 + 3 = 7
        )

        pos_a = pos_a[3:4]
        # Since pos_a is 96x96x7, only the middle (4th) 96x96 layer is needed.
        # [3:4] is taken instead of [3] since slicing keeps the third dimension [[[],[],...],...]

        row_offset = random.randrange(0, 32)  # Max offset from corner = 96 - 64 = 32
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64, col_offset:col_offset+64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset+64, col_offset:col_offset+64]).to(torch.long)

        # print("shape of pos_t: ", np.shape(pos_t))
        # ct_t is the 64x64 cropped portion of ct_a. Similarly, for pos_t.

        slice_ndx = center_irc.index

        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx


class PrepcacheLunaDataset(Dataset):  # For prepcache
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.candidateInfo_list = getCandidateInfoList()
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

        candidateInfo_tup = self.candidateInfo_list[ndx]
        getCtRawCandidate(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, (7, 96, 96))

        series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            getCtSampleSize(series_uid)
            # ct = getCt(series_uid)
            # for mask_ndx in ct.positive_indexes:
            #     build2dLungMask(series_uid, mask_ndx)

        return 0, 1 #candidate_t, pos_t, series_uid, center_t


def getCTAugmentedCandidate(  # Done AFTER cacheing so that augmentation is performed AFTER obtaining data
        augmentation_dict,
        series_uid, center_xyz, width_irc,
        use_cache=True):
    if use_cache:  # getCTRawCandidate uses cache therefore just call that
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)
    else:  # Otherwise, carry out the same steps as getCTRawCandidate but without cache decorator
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)  # Convert to flat tensor

    transform_t = torch.eye(4)  #Initialise 4D matrix

    for i in range(3):
        if 'flip' in augmentation_dict:  # Assume no strong correlation in left-right / front-back / up-down directions
            if random.random() > 0.5:  # Applied to ~ 1 in 2 calls
                transform_t[i, i] *= -1  # Flips signs of the leading diagonal

        if 'offset' in augmentation_dict:
            # May make model more robust for imperfectly annotated centre_irc values
            # random_float introduces non-int offset, introducing trilinear interpolation blurring, improving robustness
            # Pixels on border repeated (!)
            # Makes random offset in 3 dimensions, scale controlled by 'offset'
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)  # uniform random from [-1, 1]
            transform_t[i, 3] = offset_float * random_float

        if 'scaling' in augmentation_dict:
            # Pixels on border repeated if scaled down (!)
            scale_float = augmentation_dict['scaling']
            random_float = (random.random() * 2 - 1)
            transform_t[i, i] *= 1. + scale_float * random_float  # between [0, 2]

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2  # random rotation between [0, 2π]
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)
        rotation_t = torch.tensor([  # Rotation around z coordinate (CT data in (x, y, z) order)
            [c, -s,  0,  0],
            [s,  c,  0,  0],
            [0,  0,  1,  0],
            [0,  0,  0,  1],
        ])
        transform_t @= rotation_t

    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32),
        ct_t.size(),
        align_corners=False,
    )

    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode='border',
        align_corners=False,
    ).to('cpu')

    if 'noise' in augmentation_dict:  # Destructive
        noise_t = torch.rand_like(augmented_chunk)  # Uniform random [0, 1]
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    #  Implement distortions to reflect the randomness of capillary structures?


    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 ):
        self.ratio_int = ratio_int

        self.candidateInfo_list = copy.copy(getCandidateInfoList())
        # Copied so cached copy unaffected when changes are made to self.candidateInfo_list
        # changes are made below when cropping down data to validation nodules, if we are finding validation data

        if series_uid:  # Not none, then crop down candidateInfo_list to only contain candidates in uid
            self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid == series_uid]

        if isValSet_bool:  # If series_uid is part of a validation set,
            assert val_stride > 0, val_stride  # checks val_stride > 0, and  if not, returns val_stride in error
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]  # crop to data in validation set
            assert self.candidateInfo_list  # checks self.candidateInfo_list is not None (?)
        elif val_stride > 0:  # If it is not validation set (so is training data) but val_stride > 0,
            del self.candidateInfo_list[::val_stride]  # delete the validation data in series uid
            assert self.candidateInfo_list  # checks self.candidateInfo_list is not None (?)

        if sortby_str == 'random':  # Shuffle candidateInfo_list for e.g. training (DEFAULT)
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':  # Sort by series_uid if specified. Used for cacheing
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.negative_list = [nt for nt in self.candidateInfo_list if not nt.isNodule_bool]
        self.positive_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        log.info("{!r}: {} {} samples".format(
            self, len(self.candidateInfo_list), "validation" if isValSet_bool else "training",
        ))

    def __len__(self):
        if self.ratio_int:
            return 200_000  # Hardcoded to 200,000 for faster results when test training
        else:
            return len(self.candidateInfo_list)  # candidateInfo_list is list of tuples for every candidate in dataset

    def __getitem__(self, ndx):

        # candidateInfo_tup = self.candidateInfo_list[ndx]  # Old implementation before index ratio

        if self.ratio_int:  # If negative / positive ratio is specified,
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):  # If non-zero, then return non-nodule index from neg list
                neg_ndx = ndx - pos_ndx - 1
                neg_ndx %= len(self.negative_list)  # If neg_ndx < 0 or > len
                candidateInfo_tup = self.negative_list[neg_ndx]
            else:  # If zero, then return nodule index from pos list
                pos_ndx %= len(self.positive_list)  # Ensure valid index, and cycling through pos nodules
                candidateInfo_tup = self.positive_list[pos_ndx]
        else:
            candidateInfo_tup = self.candidateInfo_list[ndx]

        width_irc = (32, 48, 48)  # Fixed volume size of scan data for input

        candidate_a, center_irc = getCtRawCandidate(  # Get data array and center in IRC coords from uid, xyz data
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)     # Change format and data types of data array into tensors
        candidate_t = candidate_t.to(torch.float32)     # for PyTorch.
        candidate_t = candidate_t.unsqueeze(0)          #

        pos_t = torch.tensor(  # pos_t = [0, 1] if candidate IS a nodule, [1, 0] if NOT a nodule
            [
                    not candidateInfo_tup.isNodule_bool,
                    candidateInfo_tup.isNodule_bool
                    ],
            dtype=torch.long,
        )

        return(
            candidate_t,                    # Training sample input,
            pos_t,                          # Training sample true output,
            candidateInfo_tup.series_uid,   # series_uid,
            torch.tensor(center_irc),       # center_irc,
        )

    def shuffleSamples(self):  # Called at beginning of each epoch to ensure samples are presented in a random order
        if self.ratio_int:  # if != 0
            random.shuffle(self.negative_list)
            random.shuffle(self.positive_list)
