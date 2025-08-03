# pair_generator.py
# ------------------
# Script for generating labeled image pairs for training similarity models.
#
# Overview:
# Creates positive (same shape) and negative (different shape) image pairs
# using metadata about 3D object views.
#
# Description:
# - Reads metadata JSONL containing fields like filename, shape_id, view_id
# - Forms positive pairs by matching different views of the same shape
# - Forms negative pairs by randomly sampling views from different shapes
# - Outputs a JSONL of image pairs with labels (1 for same shape, 0 for different)
#
# Example:
#   gen = PairGen("metadata.jsonl", "pairs.jsonl", neg_samples_per_pos=2)
#   gen.run()

import json
import random
from collections import defaultdict
import os


class PairGen:

    def __init__(self, input_file, output_file, neg_samples_per_pos):
        # Args:
        #   input_file (str): Path to metadata JSONL file
        #   output_file (str): Path to save output pair JSONL
        #   neg_samples_per_pos (int): Number of negative pairs per positive
        self.INPUT_FILE = input_file
        self.OUTPUT_JSONL = output_file
        self.NEGATIVE_SAMPLES_PER_POSITIVE = neg_samples_per_pos

    def read_metadata(self, file_path):
        # Args:
        #   file_path (str): Path to JSONL metadata file
        #
        # Returns:
        #   list[dict]: All metadata entries
        items = []
        with open(file_path, 'r') as f:
            for line in f:
                items.append(json.loads(line))
        return items

    def group_by_shape_id(self, items):
        # Args:
        #   items (list[dict]): Metadata entries
        #
        # Returns:
        #   dict[str, list[dict]]: Mapping from shape_id to image entries
        shape_dict = defaultdict(list)
        for item in items:
            shape_dict[item['shape_id']].append(item)
        return shape_dict

    def generate_positive_pairs(self, shape_dict):
        # Args:
        #   shape_dict (dict): Mapping of shape_id to images
        #
        # Returns:
        #   list[dict]: Positive image pairs (same shape)
        positives = []
        for shape_id, imgs in shape_dict.items():
            if len(imgs) < 2:
                continue
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    if imgs[i]['view_id'] != imgs[j]['view_id']:
                        positives.append({
                            "img1": imgs[i]['filename'],
                            "img2": imgs[j]['filename'],
                            "label": 1
                        })
        return positives

    def generate_negative_pairs(self, items, positives, count_per_positive=1):
        # Args:
        #   items (list[dict]): All metadata entries
        #   positives (list[dict]): List of positive pairs
        #   count_per_positive (int): Number of negatives per positive
        #
        # Returns:
        #   list[dict]: Negative image pairs (different shapes)
        negatives = []
        shape_id_to_items = defaultdict(list)
        for item in items:
            shape_id_to_items[item['shape_id']].append(item)

        shape_ids = list(shape_id_to_items.keys())

        for _ in range(len(positives) * count_per_positive):
            shape_id1, shape_id2 = random.sample(shape_ids, 2)
            img1 = random.choice(shape_id_to_items[shape_id1])
            img2 = random.choice(shape_id_to_items[shape_id2])
            negatives.append({
                "img1": img1['filename'],
                "img2": img2['filename'],
                "label": 0
            })
        return negatives

    def save_jsonl(self, pairs, output_file):
        # Args:
        #   pairs (list[dict]): List of pair records to save
        #   output_file (str): Destination JSONL path
        with open(output_file, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')

    def run(self):
        # Main method to generate and save labeled pairs
        rel_in = os.path.relpath(self.INPUT_FILE)
        print(f"Reading metadata from {rel_in}...")
        items = self.read_metadata(self.INPUT_FILE)
        print(f"Found {len(items)} metadata entries.")

        shape_dict = self.group_by_shape_id(items)
        positives = self.generate_positive_pairs(shape_dict)
        print(f"Generated {len(positives)} positive pairs.")

        negatives = self.generate_negative_pairs(items, positives, self.NEGATIVE_SAMPLES_PER_POSITIVE)
        print(f"Generated {len(negatives)} negative pairs.")

        all_pairs = positives + negatives
        random.shuffle(all_pairs)

        rel_out = os.path.relpath(self.OUTPUT_JSONL)
        print(f"Saving {len(all_pairs)} pairs to {rel_out}...")
        self.save_jsonl(all_pairs, self.OUTPUT_JSONL)
        print("Done.")
