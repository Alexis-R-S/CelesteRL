import torch

from celeste_agents.base_inferencer import BaseInferencer
from celeste_agents.cnn_over_tiles.celeste_ai import CelesteAI

class Inferencer (BaseInferencer):
    ACTION_THRESHOLD = 0.5  # Treshold for deciding whether to perform an action based on the model's output

    def __init__(self, checkpoint_path=None):
        self.model = CelesteAI()
        if checkpoint_path:
            self.load_model(checkpoint_path)

    def next_state(self, player_state):
        return self._decode_action(self.model.map_and_forward(player_state))

    def load_model(self, path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model"])

    def _decode_action(self, output_tensor):
        logits = output_tensor[0]
        """Decode actions from the model's output tensor."""
        # - 3 logits for left/right/none,
        # - 3 logits for up/down/none
        # - 1 logit for jump
        # - 1 logit for dash
        # - 1 logit for grab
        horizontal_action = logits[0:3]  # Logits for left, right, none
        vertical_action = logits[3:6]    # Logits for up, down, none
        return [
            1 if (horizontal_action[1]>horizontal_action[0] and horizontal_action[1]>horizontal_action[2]) else 0,  # right
            1 if (horizontal_action[0]>=horizontal_action[1] and horizontal_action[0]>=horizontal_action[2]) else 0,  # left
            1 if (vertical_action[0]>vertical_action[1] and vertical_action[0]>vertical_action[2]) else 0,    # up
            1 if (vertical_action[1]>=vertical_action[0] and vertical_action[1]>=vertical_action[2]) else 0,     # down
            1 if logits[6] > self.ACTION_THRESHOLD else 0,   # jump
            1 if logits[7] > self.ACTION_THRESHOLD else 0,  # dash
            1 if logits[8] > self.ACTION_THRESHOLD else 0   # grab
        ]