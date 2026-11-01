import importlib

from celeste_agents.base_ai_trainer import BaseAiTrainer
from controller.base_controller import BaseController
from celeste_agents.agents_list import AGENTS

class TrainingController(BaseController):
    def __init__(self, agent_name, load_checkpoint_path=None, save_checkpoint_folder=None):
        # Load trainer and create instance
        self.trainer: BaseAiTrainer = self._load_trainer(agent_name)(load_checkpoint_path, save_checkpoint_folder)

    def update(self, player_state):
        return self.trainer.next_state(player_state)

    def end_sequence(self, player_state):
        return self.trainer.end_episode(player_state)
        
    def end_session(self, save_checkpoint_path):
        return self.trainer.save_model(save_checkpoint_path)

    def _load_trainer(self, agent_name) -> BaseAiTrainer:
        agent_info = AGENTS.get(agent_name)
        if not agent_info:
            raise ValueError(f"Unknown agent {agent_name}")

        trainer_module = agent_info.get('module')
        trainer_file = agent_info.get('trainer')
        trainer_class = agent_info.get('trainer_class')

        try:
            module = importlib.import_module(f"celeste_agents.{trainer_module}.{trainer_file}")
            trainer_class = getattr(module, trainer_class)
            return trainer_class
        except ImportError as e:
            raise ImportError(f"Could not import trainer {trainer_class} from module {trainer_module}.{trainer_file}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Trainer class {trainer_class} not found in module {trainer_module}.{trainer_file}: {e}")