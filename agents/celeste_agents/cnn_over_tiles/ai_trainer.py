from celeste_agents.base_ai_trainer import BaseAiTrainer
from celeste_agents.cnn_over_tiles.celeste_ai import CelesteAI
from celeste_agents.cnn_over_tiles.reward_tracker import RewardTracker

import torch
import os

class AiTrainer (BaseAiTrainer):
    batch_size = 10     # Number of episodes to accumulate before performing a training step

    def __init__(self, checkpoint_path=None):
        self.model = CelesteAI()
        self.reward_tracker = RewardTracker()
        self.log_probs = []     # Log probs of actions taken during the current episode

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )

        if checkpoint_path:
            self.load_model(checkpoint_path)

        self.batch_losses = []  # Store losses for each episode in the batch

    def next_state(self, player_state):
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

    def end_episode(self, player_state):
        self.reward_tracker.add_reward(player_state)  # Add final reward without log_prob
        self.reward_tracker.compound_end_reward(player_state)  # Adjust the final reward based on the outcome
        loss = self.reward_tracker.calculate_loss(self.log_probs)
        self.batch_losses.append(loss)

        # Perform training step if we have accumulated enough episodes
        if len(self.batch_losses) >= self.batch_size:
            self._perform_training_step()

        self.log_probs.clear()
        self.reward_tracker.reset()  # Reset the reward tracker for the next sequence

    def end_session(self, save_checkpoint_path=None):
        if save_checkpoint_path:
            self.save_model(save_checkpoint_path)

    def save_model(self, checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, checkpoint_path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def _perform_training_step(self):
        average_loss = torch.stack(self.batch_losses).mean()    # Calculate the average loss over the batch

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        average_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()
        self.batch_losses.clear()   # Clear the batch losses for the next batch

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