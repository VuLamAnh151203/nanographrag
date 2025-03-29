import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="A simple script with arguments")

# Add an argument
parser.add_argument("--name", type=str, help="Your name", required=True)
parser.add_argument("--age", type=int, help="Your age", default=25)

# Parse the arguments
args = parser.parse_args()

# Use the arguments
print(f"Hello, {args.name}! You are {args.age} years old.")
