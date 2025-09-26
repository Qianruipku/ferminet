import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        print("Usage: python print_checkpoint.py <checkpoint_file>")
        sys.exit(1)

    checkpoint_file = sys.argv[1]

    try:
        data = np.load(checkpoint_file, allow_pickle=True)
        print("Checkpoint data:")
        for key in data.files:
            print(f"{key}: {data[key]}")
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()