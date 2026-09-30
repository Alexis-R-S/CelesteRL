from CelestePythonInterface import SessionData
from numpy import cos, sin
import torch
import torch.nn as nn
import torch.nn.functional as F

class CelesteAI(nn.Module):
    OCCUPANCY_MAP_X = 31
    OCCUPANCY_MAP_Y = 31
    OCCUPANCY_MAP_Z = 5
    ACTION_THRESHOLD = 0.5

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.OCCUPANCY_MAP_Z,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        # CNN gives 32 features
        # + 12 state variables
        self.fc = nn.Sequential(
            nn.Linear(32 + 12, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )
        # 9 outputs :
        # - 3 logits for left/right/none,
        # - 3 logits for up/down/none
        # - 1 logit for jump
        # - 1 logit for dash
        # - 1 logit for grab

    def forward(self, occupancy_map, state_variables):
        cnn_features = self.cnn(occupancy_map)
        cnn_features = torch.flatten(cnn_features, 1)   # Convert: (batch, 32, 1, 1) into: (batch, 32)

        combined = torch.cat(
            (cnn_features, state_variables),
            dim=1
        )

        output = self.fc(combined)
        return output

    def map_and_forward(self, player_state):
        state_variables, occupancy_map = self._map_player_state(player_state)
        return self.forward(occupancy_map, state_variables)

    def _map_player_state(self, player_state):
        # Input mapping
        tile_size = player_state[SessionData.TILE_SIZE.value]
        rel_pos_x = (player_state[SessionData.X_POSITION.value] - player_state[SessionData.X_OCCUPANCY_MAP_POSITION.value]) / tile_size -15
        rel_pos_y = (player_state[SessionData.Y_POSITION.value] - player_state[SessionData.Y_OCCUPANCY_MAP_POSITION.value]) / tile_size -15

        # Build input tensor
        # Shape : (batch=1, length=12)
        input_tensor = torch.tensor([
            player_state[SessionData.X_VELOCITY.value],
            player_state[SessionData.Y_VELOCITY.value],
            sin(rel_pos_x),
            cos(rel_pos_x),
            sin(rel_pos_y),
            cos(rel_pos_y),
            player_state[SessionData.ON_GROUND.value],
            player_state[SessionData.CAN_DASH.value],
            player_state[SessionData.CAN_SECOND_DASH.value],
            player_state[SessionData.STAMINA.value],
            player_state[SessionData.X_DISTANCE_TO_OBJECTIVE.value],
            player_state[SessionData.Y_DISTANCE_TO_OBJECTIVE.value]
        ], dtype=torch.float32).unsqueeze(0)

        # Build occupancy map tensor
        # Shape : (batch=1, channels=3, height=31, width=31)
        occupancy = torch.tensor([
            player_state[SessionData.OCCUPANCY_MAP.value:SessionData.OCCUPANCY_MAP.value+self.OCCUPANCY_MAP_X*self.OCCUPANCY_MAP_Y]
        ], dtype=torch.long).reshape(self.OCCUPANCY_MAP_X, self.OCCUPANCY_MAP_Y)
        occupancy = F.one_hot(occupancy, num_classes=self.OCCUPANCY_MAP_Z).permute(2, 0, 1).unsqueeze(0).float()

        return (input_tensor, occupancy)
    