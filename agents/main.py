from celeste_connector import CelesteConnector
from controller.inference_controller import InferenceController
from controller.training_controller import TrainingController
from controller.user_controller import UserController
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser(description="Celeste AI Controller")
parser.add_argument("--train", action="store_true", help="Run in training mode")
parser.add_argument("--user", action="store_true", help="Run in user mode")
parser.add_argument("--limit", type=int, default=100, help="Number of batches to run (default: 100)")
parser.add_argument("--save", type=str, default="model.pth", help="Checkpoint name to save (default: model.pth)")
parser.add_argument("--load", type=str, default=None, help="Checkpoint name to load (default: None)")
args = parser.parse_args()

checkpoint_load_path = "checkpoints/" + args.load if args.load else None

if args.train:
    print("Launching AI in training mode")
    controller = TrainingController(checkpoint_path=checkpoint_load_path)
elif args.user:
    print("Launching AI in user mode")
    controller = UserController()
else:
    print("Launching AI in inference mode")
    controller = InferenceController(checkpoint_path=checkpoint_load_path)

if args.limit is None:
    print("No limit specified, running indefinitely.")
    batch_limit = float('inf')  # Run indefinitely
else:
    print(f"Running for {args.limit} batches.")
    batch_limit = args.limit

celeste_conn = CelesteConnector()
celeste_conn.connect()

celeste_conn.set_agent(controller.update)

for _ in range(batch_limit):    # Control goes back here after each player death
    final_state = celeste_conn.run()
    controller.end_sequence(final_state)

if args.save:
    checkpoint_path = "checkpoints/" + args.save
else:
    checkpoint_path = "checkpoints/default.pth"

print(f"Saving model to {checkpoint_path}")
controller.end_session(save_checkpoint_path=checkpoint_path)