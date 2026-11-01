from os import path

from celeste_connector import CelesteConnector
from controller.inference_controller import InferenceController
from controller.training_controller import TrainingController
from controller.user_controller import UserController

import argparse
from dotenv import load_dotenv
import os

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Celeste AI Controller")
parser.add_argument("--train", action="store_true", help="Run in training mode")
parser.add_argument("--user", action="store_true", help="Run in user mode")
parser.add_argument("--limit", type=int, default=100, help="Number of episodes to run (default: 100)")
parser.add_argument("--save", type=str, default="default", help="Folder name to save the model to (default: default)")
parser.add_argument("--load", type=str, default=None, help="Checkpoint name to load (default: None)")
parser.add_argument("--choosing", type=str, default="random", help="Method to choose rooms")
args = parser.parse_args()

checkpoint_load_path = "checkpoints/" + args.load if args.load else None

if args.save:
    checkpoint_folder = "checkpoints/" + args.save
else:
    checkpoint_folder = "checkpoints/default"

if args.train:
    print("Launching AI in training mode")
    agent_name = os.getenv('AGENT')
    if not agent_name:
        raise RuntimeError("AGENT is not defined in .env")
    controller = TrainingController(
        agent_name=agent_name,
        load_checkpoint_path=checkpoint_load_path,
        save_checkpoint_folder=checkpoint_folder
    )

elif args.user:
    print("Launching AI in user mode")
    controller = UserController()
else:
    print("Launching AI in inference mode")
    agent_name = os.getenv('AGENT')
    if not agent_name:
        raise RuntimeError("AGENT is not defined in .env")
    controller = InferenceController(agent_name=agent_name, checkpoint_path=checkpoint_load_path)

if args.limit is None:
    print("No limit specified, running indefinitely.")
    batch_limit = float('inf')  # Run indefinitely
else:
    print(f"Running for {args.limit} batches.")
    batch_limit = args.limit

celeste_conn = CelesteConnector(room_choosing_mode=args.choosing)
celeste_conn.connect()

celeste_conn.set_agent(controller.update)

for _ in range(batch_limit):    # Control goes back here after each player death
    final_state = celeste_conn.run()
    controller.end_sequence(final_state)
    celeste_conn.next_room()

# Saving the model after training
checkpoint_path = path.join(checkpoint_folder, 'model.pth')
print(f"Saving model to {checkpoint_path}")
controller.end_session(save_checkpoint_path=checkpoint_path)