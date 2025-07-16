from data.pair_generator import PairGen
import os
# Code to run pair generator from root. Run python -m experiments.test to run this file from root

def main():
    # Compute project root automatically
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Correct absolute paths
    input_file = os.path.join(project_root, "data", "example", "training", "metadata.jsonl")
    output_file = os.path.join(project_root, "experiments", "pairs.jsonl")
    num_neg = 1

    pair_gen = PairGen(input_file, output_file, num_neg)
    pair_gen.run()

if __name__ == "__main__":
    main()