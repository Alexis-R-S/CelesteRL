import importlib

from controller.base_controller import BaseController
from celeste_agents.agents_list import AGENTS

class InferenceController(BaseController):
    def __init__(self, agent_name, checkpoint_path=None):
        self.inferencer = self._load_inferencer(agent_name)(checkpoint_path)

    def update(self, player_state):
        return self.inferencer.next_state(player_state)

    def end_sequence(self, player_state):
        pass

    def end_session(self, save_checkpoint_path=None):
        pass

    def _load_inferencer(self, agent_name):
        agent_info = AGENTS.get(agent_name)
        if not agent_info:
            raise ValueError(f"Unknown agent {agent_name}")

        inference_module = agent_info.get('module')
        inference_file = agent_info.get('inferencer')
        inference_class = agent_info.get('inferencer_class')

        try:
            module = importlib.import_module(f"celeste_agents.{inference_module}.{inference_file}")
            return getattr(module, inference_class)
        except ImportError as e:
            raise ImportError(f"Could not import trainer {inference_class} from module {inference_module}.{inference_file}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Trainer class {inference_class} not found in module {inference_module}.{inference_file}: {e}")
