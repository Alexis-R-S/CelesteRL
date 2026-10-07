from abc import ABC, abstractmethod


class BaseController(ABC):
    @abstractmethod
    def update(self, player_state):
        """Met à jour les actions du joueur en fonction de l'état actuel.
        Args:
            player_state: L'état actuel du joueur fourni par l'interface de communication. Voir SessionData pour les détails sur la structure de player_state.
        Returns:
            Une liste de 7 inputs (floats) représentant les actions à effectuer. Chaque float doit être dans la plage [0, 1]. Si la valeur est supérieure à 0.5, l'action correspondante sera effectuée.
            Voir SessionData.Inputs pour les détails sur la signification de chaque input.
        """
        pass

    @abstractmethod
    def end_sequence(self, player_state):
        """Méthode appelée à la fin d'une séquence de jeu (par exemple, après la mort du joueur).
        Args:
            player_state: L'état final du joueur fourni par l'interface de communication. Voir SessionData pour les détails sur la structure de player_state.
        """
        pass

    @abstractmethod
    def end_session(self, save_checkpoint_path=None):
        pass