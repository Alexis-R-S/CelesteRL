from abc import ABC, abstractmethod

class BaseAiTrainer (ABC):
    @abstractmethod
    def __init__(self, checkpoint_path=None):
        pass

    @abstractmethod
    def next_state(self, player_state):
        pass

    @abstractmethod
    def end_episode(self, player_state):
        pass

    @abstractmethod
    def end_session(self, save_checkpoint_path=None):
        pass