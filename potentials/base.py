from abc import ABC, abstractmethod


class Potential(ABC):
    @abstractmethod
    def drift(self, x): # Force
        """
        :param x: [N x 3]
        :return:
        """
        pass