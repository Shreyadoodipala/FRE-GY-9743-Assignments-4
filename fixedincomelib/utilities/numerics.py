import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def _segment_ranges(self):
        if len(self.axis1_) == 1:
            return [(-np.inf, np.inf, 0)]

        ranges = [(-np.inf, self.axis1_[0], 0)]
        for i in range(1, len(self.axis1_)):
            ranges.append((self.axis1_[i - 1], self.axis1_[i], i))
        ranges.append((self.axis1_[-1], np.inf, len(self.axis1_) - 1))
        return ranges

    @staticmethod
    def _overlap_length(start_x: float, end_x: float, lower: float, upper: float) -> float:
        if end_x < start_x:
            return -Interpolator1DPCP._overlap_length(end_x, start_x, lower, upper)

        lower_bound = start_x if np.isneginf(lower) else max(start_x, lower)
        upper_bound = end_x if np.isposinf(upper) else min(end_x, upper)
        return max(0.0, upper_bound - lower_bound)

    def interpolate(self, x: float) -> float:
        x_arr = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.axis1_, x_arr, side='left')
        idx = np.clip(idx, 0, self.length_ - 1)
        values = self.values_[idx]
        return float(values.item()) if np.ndim(x_arr) == 0 else values

    def integrate(self, start_x: float, end_x: float) -> float:
        if start_x == end_x:
            return 0.0
        if end_x < start_x:
            return -self.integrate(end_x, start_x)

        result = 0.0
        for lower, upper, node_idx in self._segment_ranges():
            overlap = self._overlap_length(start_x, end_x, lower, upper)
            result += self.values_[node_idx] * overlap
        return float(result)

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        flat_x = np.atleast_1d(x_arr)
        idx = np.searchsorted(self.axis1_, flat_x, side='left')
        idx = np.clip(idx, 0, self.length_ - 1)

        grad = np.zeros((flat_x.size, self.length_), dtype=float)
        grad[np.arange(flat_x.size), idx] = 1.0
        return grad[0] if x_arr.ndim == 0 else grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        if start_x == end_x:
            return np.zeros(self.length_, dtype=float)
        if end_x < start_x:
            return -self.gradient_of_integrated_value_wrt_ordinate(end_x, start_x)

        grad = np.zeros(self.length_, dtype=float)
        for lower, upper, node_idx in self._segment_ranges():
            grad[node_idx] += self._overlap_length(start_x, end_x, lower, upper)
        return grad



class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')