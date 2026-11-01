import random

from CelestePythonInterface import SocketInterface, SessionParameters, SocketServer

from data.level_selector import LevelSelector

class CelesteConnector:
    def __init__(self, chapters=None, sides=None, timeout_seconds=10, objective_x=282, objective_y=-24):
        """Initialise le connecteur Celeste avec les paramètres de session spécifiés.
        Args:
            chapters (int): Les chapitres autorisés (clés de la zone)
            sides (int): Les sides de chapitres autorisés (modes du chapitre)
            timeout_seconds (int): Le temps maximum de la session en secondes.
            objective_x (int): La coordonnée X de l'objectif.
            objective_y (int): La coordonnée Y de l'objectif.
        """
        level_selector = LevelSelector()
        self.rooms_pool = level_selector.get_level_pool(chapters, sides)

        self.params = SessionParameters()
        self.params.TimeoutSeconds = timeout_seconds
        self.params.ObjectiveXCoordinate = objective_x
        self.params.ObjectiveYCoordinate = objective_y

        self.randomize_level()

        self.interface = None

    def connect(self):
        """Établit la connexion avec le jeu Celeste et initialise l'interface de communication."""

        print("En attente de connexion avec Celeste...")
        server = SocketServer()
        client_socket, addr = server.await_connection()
        print(f"Connecté ! ({addr})")

        def no_agent(player_state):     # Agent function to do nothing (default function)
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.interface = SocketInterface(client_socket, no_agent, self.params)

    def set_agent(self, agent_function):
        """Définit la fonction de l'agent qui sera appelée à chaque frame.
        Args:
            agent_function (function): La fonction de l'agent qui prend en entrée l'état du joueur et retourne les actions à effectuer.
        """
        self.interface.agent = agent_function

    def run(self):
        """Exécute la session de jeu avec l'agent défini.
        S'arrête lorsque le joueur meurt ou que le temps imparti est écoulé.
        Returns:
            final_state: L'état final de la session.
        """
        final_state = self.interface.run()

        return final_state

    def randomize_level(self):
        rdm_chap, rdm_side, rdm_room = random.choice(self.rooms_pool)
        self.params.AreaKey = rdm_chap
        self.params.AreaMode = rdm_side
        self.params.Level = rdm_room
