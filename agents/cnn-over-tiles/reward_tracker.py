from CelestePythonInterface import SessionData
import torch

class RewardTracker:
    GAMMA = 0.99  # Discount factor for future rewards

    def __init__(self):
        self.rewards = []
        self.previous_distance = None

    def add_reward(self, player_state):
        distance = self._calculate_distance(player_state)
        if self.previous_distance is None:
            self.previous_distance = distance
            return 0

        reward = self.previous_distance - distance
        self.rewards.append(reward)
        self.previous_distance = distance

    def _calculate_distance(self, player_state):
        x_distance = player_state[SessionData.X_DISTANCE_TO_OBJECTIVE.value]
        y_distance = player_state[SessionData.Y_DISTANCE_TO_OBJECTIVE.value]
        return (x_distance ** 2 + y_distance ** 2) ** 0.5

    def calculate_loss(self, log_probs):
        returns = self._calculate_returns()
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # Normalize returns to have mean 0 and std 1

        loss = torch.stack([
            -log_prob * G
            for log_prob, G in zip(self.log_probs, returns)
        ]).sum()

        return loss

    def reset(self):
        self.rewards.clear()
        self.previous_distance = None

    def _calculate_returns(self):
        returns = []
        G = 0.0

        for reward in reversed(self.rewards):
            G = reward + self.GAMMA * G  # Assuming a discount factor of 0.99
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32)