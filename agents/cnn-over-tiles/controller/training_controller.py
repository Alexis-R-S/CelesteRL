from CelestePythonInterface import SessionData

from celeste_ai import CelesteAI
from controller.base_controller import BaseController
from reward_tracker import RewardTracker

import torch

class TrainingController(BaseController):
    def __init__(self):
        self.model = CelesteAI()
        self.reward_tracker = RewardTracker()
        self.log_probs = []

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )

    def update(self, player_state):
        """Met à jour les actions du joueur en fonction de l'état actuel.
        Args:
            player_state: L'état actuel du joueur fourni par l'interface de communication. Voir SessionData pour les détails sur la structure de player_state.
        Returns:
            Une liste de 7 inputs (floats) représentant les actions à effectuer. Chaque float doit être dans la plage [0, 1]. Si la valeur est supérieure à 0.5, l'action correspondante sera effectuée.
            Voir SessionData.Inputs pour les détails sur la signification de chaque input.
        """
        output_tensor = self.model.map_and_forward(player_state)
        actions, log_prob = self._decode_action(output_tensor)
        self.log_probs.append(log_prob)

        self.reward_tracker.add_reward(player_state)

        return actions

    def end_sequence(self, player_state):
        self.reward_tracker.add_reward(player_state)  # Add final reward without log_prob
        loss = self.reward_tracker.calculate_loss(self.log_probs)
        
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        self.log_probs.clear()
        self.reward_tracker.reset()  # Reset the reward tracker for the next sequence

    def _decode_action(self, output_tensor):
            logits = output_tensor[0]

            # Create distributions for each action
            distributions = [
                torch.distributions.Categorical(logits=logits[0:3]),  # Logits for left, right, none
                torch.distributions.Categorical(logits=logits[3:6]),    # Logits for up, down, none
                torch.distributions.Bernoulli(logits=logits[6]),   # Logit for jump
                torch.distributions.Bernoulli(logits=logits[7]),  # Logit for dash
                torch.distributions.Bernoulli(logits=logits[8])   # Logit for grab
            ]
    
            # Sample actions from the distributions
            samples = [dist.sample() for dist in distributions]
    
            # Calculate log probabilities
            log_prob = sum([dist.log_prob(sample) for dist, sample in zip(distributions, samples)])
    
            # Convert samples to the required action format
            actions = [
                1 if samples[0] == 1 else 0,  # right
                1 if samples[0] == 0 else 0,  # left
                1 if samples[1] == 0 else 0,    # up
                1 if samples[1] == 1 else 0,     # down
                int(samples[2].item()),   # jump
                int(samples[3].item()),  # dash
                int(samples[4].item())   # grab
            ]
    
            return actions, log_prob