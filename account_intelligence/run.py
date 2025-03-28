import os
import sys
import argparse

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Account Intelligence --> powered by AI')
parser.add_argument('--iteration', type=int, choices=[1, 2, 3], default=1, 
                    help='Which iteration to run (1, 2, or 3)')
parser.add_argument('--mode', type=str, choices=['part1', 'part2', 'part3'], 
                    default='part1', help='Which part of the selected iteration to run')
args = parser.parse_args()

# Import and run the app
from account_intelligence_ai.app import create_demo

if __name__ == "__main__":
    demo = create_demo(iteration=args.iteration, mode_str=args.mode)
    demo.launch()
