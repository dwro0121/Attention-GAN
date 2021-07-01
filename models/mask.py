import numpy as np
import random
import math
from collections import defaultdict
import copy


def allow_non_square(fn):
    def wrap(self, nI, **kwargs):
        nO = kwargs.pop("nO", None)
        if nO is None:
            nO = nI
        return fn(self, nI, nO, **kwargs)

    return wrap


def numpy(fn):
    def wrap(self, *args, **kwargs):
        indices = fn(self, *args, **kwargs)
        tensor = np.array(indices, dtype=np.float32)
        return tensor

    return wrap


def disallow_downsampling(fn):
    def wrap(self, nI, **kwargs):
        nO = kwargs.pop("nO", None)
        if nO is None:
            nO = nI
        if nO < nI:
            raise ValueError("Downsampling not supported.")
        return fn(self, nI, nO, **kwargs)

    return wrap


def disallow_non_square(fn):
    def wrap(self, nI, **kwargs):
        nO = kwargs.pop("nO", None)
        if nO is not None and nO != nI:
            raise ValueError("Non square masks not supported")
        return fn(self, nI, **kwargs)

    return wrap


def compute_stride(fn):
    def wrap(self, nL, nO=None, **kwargs):
        stride = kwargs.pop("stride", None)
        if stride is None:
            stride = math.floor(math.sqrt(nL))
        return fn(self, nL, nO=nO, stride=stride, **kwargs)

    return wrap


def _2d_to_1d(x, y, cols):
    return x * cols + y


def esa(rols, cols):
    list_2d = []
    for i in range(rols):
        for j in range(cols):
            list_2d.append([i, j])
    list_2d = sorted(list_2d, key=lambda x: x[0] + x[1])
    return list_2d


class SparseMask:
    @classmethod
    def get_maps(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    @disallow_non_square
    @numpy
    def get_equal_maps_from_1d(self, input_2d):
        rows, cols = input_2d
        esa_list = esa(rows, cols)
        maps = self.get_maps(rows * cols)
        equal_maps = []
        for input_node, output_node in maps:
            # print('input_node = {}, output_node = {}'.format(input_node,output_node))

            input_x, input_y = esa_list[int(input_node)]
            output_x, output_y = esa_list[int(output_node)]

            input_esa_1d = _2d_to_1d(input_x, input_y, cols)
            output_esa_1d = _2d_to_1d(output_x, output_y, cols)
            equal_maps.append([input_esa_1d, output_esa_1d])
        return equal_maps

    @classmethod
    @allow_non_square
    @numpy
    def get_general_maps(self, input_2d, output_2d):
        input_rols, input_cols = input_2d
        output_rols, output_cols = output_2d

        if input_rols == output_rols and input_cols == output_cols:
            return self.get_equal_maps_from_1d(input_2d)

        input_1d = input_rols * input_cols
        output_1d = output_rols * output_cols
        block_num = output_1d // input_1d
        init_maps = self.get_equal_maps_from_1d(input_2d)
        general_maps = copy.deepcopy(init_maps)
        offset_maps = np.zeros_like(init_maps)
        for _ in range(1, block_num):
            offset_maps[:, 0] += input_1d
            new_maps = copy.deepcopy(init_maps)
            new_maps += offset_maps
            general_maps = np.concatenate([general_maps, new_maps])
        return general_maps

    @classmethod
    @allow_non_square
    def get_mask(self, input_2d, output_2d, **kwargs):
        output_rols, output_cols = output_2d
        output_1d = output_rols * output_cols
        input_rows, input_cols = input_2d
        input_1d = input_rows * input_cols

        maps = self.get_general_maps(input_2d, nO=output_2d, **kwargs)
        tensor = np.zeros([output_1d, input_1d], dtype=np.float32)
        tensor[maps[:, 0].astype(int), maps[:, 1].astype(int)] = 1
        return tensor


class RTLMask_layer1(SparseMask):
    @classmethod
    @disallow_downsampling
    @compute_stride
    @numpy
    def get_maps(self, nI, nO, stride=None, **kwargs):
        maps = []
        for row in range(nO):
            lower = max(0, min(row - (row % stride), nI))
            higher = min(row + 1, nI)
            for col in range(lower, higher):
                maps.append([row, col])
        return maps


class LTRMask_layer1(SparseMask):
    @classmethod
    @disallow_downsampling
    @compute_stride
    @numpy
    def get_maps(self, nI, nO, stride=None, **kwargs):
        maps = []
        for row in range(nO):
            lower = max(0, min(row - (row % stride), nI))
            higher = min(row + 1, nI)
            for col in range(lower, higher):
                maps.append([col, row])
        return maps


class RTLMask_layer2(SparseMask):
    @classmethod
    @compute_stride
    @allow_non_square
    @numpy
    def get_maps(self, nI, nO, stride=None, overlap=1, **kwargs):
        maps = []
        row_maps = np.arange(0, nI)
        indxs = (row_maps % stride) >= (stride - overlap)
        col_maps = row_maps[indxs]

        for row in np.arange(0, nO):
            if nI == nO:
                maps.append([row, row])
            for col in col_maps:
                maps.append([row, col])
        return maps


class LTRMask_layer2(SparseMask):
    @classmethod
    @allow_non_square
    @compute_stride
    @numpy
    def get_maps(self, nI, nO, stride=None, overlap=1, **kwargs):
        maps = []
        row_maps = np.arange(0, nI)
        indxs = (row_maps % stride) >= (stride - overlap)
        col_maps = row_maps[indxs]
        col_maps = col_maps - (stride - overlap)

        for row in np.arange(0, nO):
            if nI == nO:
                maps.append([row, row])
            for col in col_maps:
                maps.append([row, col])
        return maps
