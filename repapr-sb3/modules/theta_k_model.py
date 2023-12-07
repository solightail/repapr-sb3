from .conf import Conf
from abc import ABC, abstractmethod
import numpy as np

class Algorithm(ABC):
    """ Algorithm super class """
    def __init__(self, manual: list=None) -> None:
        cfg = Conf()
        self.tones: int = cfg.tones
        self.phase_value = cfg.phase_value
        self.const_first_phase = cfg.const_first_phase
        if manual is None: 
            manual = cfg.manual
        else:
            self.manual = manual

    @abstractmethod
    def calc(self) -> np.ndarray:
        """ calc theta_k abstract method """

class Narahashi(Algorithm):
    """ Narahashi Algorithm """
    def calc(self) -> np.ndarray:
        indexes: np.ndarray = np.arange(self.tones)
        theta_k_bins: np.ndarray = ((indexes)*(indexes - 1)) / (2*(self.tones - 1))
        if self.const_first_phase is True:
            theta_k_bins = theta_k_bins + self.phase_value
        theta_k_bins = np.mod(theta_k_bins, 1)
        return np.array(theta_k_bins, dtype='float32')

class Newman(Algorithm):
    """ Newman Algorithm """
    def calc(self) -> np.ndarray:
        indexes: np.ndarray = np.arange(self.tones)
        theta_k_bins: np.ndarray = (((indexes-1)**2) / (2*self.tones))
        if self.const_first_phase is True:
            theta_k_bins = theta_k_bins - theta_k_bins[0] + self.phase_value
        theta_k_bins = np.mod(theta_k_bins, 1)
        return np.array(theta_k_bins, dtype='float32')

class Unify(Algorithm):
    """ Unify all phases """
    def calc(self) -> np.ndarray:
        theta_k_bins: np.ndarray = np.full(self.tones, self.phase_value)
        return np.array(theta_k_bins, dtype='float32')

class Random(Algorithm):
    """ Random theta_k_bins """
    def calc(self) -> np.ndarray:
        if self.const_first_phase is True:
            theta_k_bins: np.ndarray = np.insert(np.random.rand(self.tones-1), 0, self.phase_value)
        else:
            theta_k_bins: np.ndarray = np.random.rand(self.tones)
        return np.array(theta_k_bins, dtype='float32')

class Manual(Algorithm):
    """ Manual theta_k_bins """
    def calc(self) -> np.ndarray:
        return np.array(self.manual, dtype='float32')


class AContext:
    """ Algorithm Context """
    def __init__(self, strategy: Algorithm) -> None:
        self._strategy = strategy
        self.theta_k_bins: np.ndarray = None

    def calc_algo(self) -> np.ndarray:
        """ Calculation each algorithm """
        if self.theta_k_bins is None:
            self.theta_k_bins = self._strategy.calc()
        return self.theta_k_bins
