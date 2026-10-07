from abc import ABC, abstractmethod

class BaseInferencer (ABC):
    @abstractmethod
    def __init__(self, checkpoint_path=None):
        pass

    @abstractmethod
    def next_state(self, player_state):
        pass

    