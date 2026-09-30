from celeste_connector import CelesteConnector
from controller.inference_controller import InferenceController
from controller.training_controller import TrainingController
from controller.user_controller import UserController
from CelestePythonInterface import SessionData
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Celeste AI Controller")
parser.add_argument("--train", action="store_true", help="Run in training mode")
parser.add_argument("--user", action="store_true", help="Run in user mode")
args = parser.parse_args()

if args.train:
    print("Launching AI in training mode")
    Controller = TrainingController
elif args.user:
    print("Launching AI in user mode")
    Controller = UserController
else:
    print("Launching AI in inference mode")
    Controller = InferenceController

celeste_conn = CelesteConnector()
celeste_conn.connect()

controller = Controller()
celeste_conn.set_agent(controller.update)

while True:     # Control goes back here after each player death
    final_state = celeste_conn.run()
    controller.end_sequence(final_state)