from CelestePythonInterface import SessionData
import torch

from cnn_over_tiles.celeste_ai import CelesteAI
from controller.base_controller import BaseController

class InferenceController(BaseController):
    def __init__(self, checkpoint_path=None):
        self.model = CelesteAI()
        if checkpoint_path:
            self.load_model(checkpoint_path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])

    def update(self, player_state):
        """Met à jour les actions du joueur en fonction de l'état actuel.
        Args:
            player_state: L'état actuel du joueur fourni par l'interface de communication. Voir SessionData pour les détails sur la structure de player_state.
        Returns:
            Une liste de 7 inputs (floats) représentant les actions à effectuer. Chaque float doit être dans la plage [0, 1]. Si la valeur est supérieure à 0.5, l'action correspondante sera effectuée.
            Voir SessionData.Inputs pour les détails sur la signification de chaque input.
        """
        actions = self.decode_action(self.model.map_and_forward(player_state))
        return actions

    def end_sequence(self, player_state):
        pass

    def end_session(self, save_checkpoint_path=None):
        pass

    def decode_action(self, output_tensor):
        """Décodage des actions à partir du tenseur de sortie du modèle."""
        # - 3 logits for left/right/none,
        # - 3 logits for up/down/none
        # - 1 logit for jump
        # - 1 logit for dash
        # - 1 logit for grab
        horizontal_action = output_tensor[0:3]  # Logits for left, right, none
        vertical_action = output_tensor[3:6]    # Logits for up, down, none
        return [
            1 if horizontal_action == 1 else 0,  # right
            1 if horizontal_action == 0 else 0,  # left
            1 if vertical_action == 0 else 0,    # up
            1 if vertical_action == 1 else 0,     # down
            1 if output_tensor[6] > self.ACTION_THRESHOLD else 0,   # jump
            1 if output_tensor[7] > self.ACTION_THRESHOLD else 0,  # dash
            1 if output_tensor[8] > self.ACTION_THRESHOLD else 0   # grab
        ]
