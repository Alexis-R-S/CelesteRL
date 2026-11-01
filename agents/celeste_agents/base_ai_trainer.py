from abc import ABC, abstractmethod

class BaseAiTrainer (ABC):
    @abstractmethod
    def __init__(self, load_checkpoint_path=None, save_checkpoint_folder=None):
        pass

    @abstractmethod
    def next_state(self, player_state):
        pass

    @abstractmethod
    def end_episode(self, player_state):
        pass

    @abstractmethod
    def save_model(self, checkpoint_path):
        pass