from CelestePythonInterface import SessionData

from celeste_ai import CelesteAI
from controller.base_controller import BaseController

class InferenceController(BaseController):
    def __init__(self):
        self.model = CelesteAI()

    def update(self, player_state):
        """Met à jour les actions du joueur en fonction de l'état actuel.
        Args:
            player_state: L'état actuel du joueur fourni par l'interface de communication. Voir SessionData pour les détails sur la structure de player_state.
        Returns:
            Une liste de 7 inputs (floats) représentant les actions à effectuer. Chaque float doit être dans la plage [0, 1]. Si la valeur est supérieure à 0.5, l'action correspondante sera effectuée.
            Voir SessionData.Inputs pour les détails sur la signification de chaque input.
        """
        actions = self.model.decode_action(self.model.map_and_forward(player_state))
        return actions

    def end_sequence(self, player_state):
        pass