from abc import ABC, abstractmethod
import numpy as np

class Algorithm(ABC):
    """ Algorithm super class """
    def __init__(self, tones: int, unify_value=None, manual=None) -> None:
        self.tones: int = tones
        if unify_value is not None: self.unify_value = unify_value
        if manual is not None: self.manual = manual

    @abstractmethod
    def calc(self) -> np.ndarray:
        """ calc theta_k abstract method """

class Narahashi(Algorithm):
    """ Narahashi Algorithm """
    def calc(self) -> np.ndarray:
        indexes: np.ndarray = np.arange(self.tones)
        theta_k_bins: np.ndarray = (((indexes)*(indexes - 1)) / (2*(self.tones - 1))) % 1
        return np.array(theta_k_bins, dtype='float32')

class Newman(Algorithm):
    """ Newman Algorithm """
    def calc(self) -> np.ndarray:
        indexes: np.ndarray = np.arange(self.tones)
        theta_k_bins: np.ndarray = (((indexes-1)**2) / (2*self.tones)) % 1
        return np.array(theta_k_bins, dtype='float32')

class Unify(Algorithm):
    """ Unify all phases """
    def calc(self) -> np.ndarray:
        theta_k_bins: np.ndarray = np.full(self.tones, self.unify_value)
        return np.array(theta_k_bins, dtype='float32')

class Random(Algorithm):
    """ Random theta_k_bins """
    def calc(self) -> np.ndarray:
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
