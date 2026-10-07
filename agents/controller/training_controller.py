from controller.base_controller import BaseController
from celeste_agents.cnn_over_tiles.ai_trainer import AiTrainer

class TrainingController(BaseController):
    def __init__(self, checkpoint_path=None):
        self.trainer = AiTrainer(checkpoint_path)

    def update(self, player_state):
        return self.trainer.next_state(player_state)

    def end_sequence(self, player_state):
        return self.trainer.end_episode(player_state)
        
    def end_session(self, save_checkpoint_path):
        return self.trainer.end_session(save_checkpoint_path)
