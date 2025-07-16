# pair_generator.py
# ------------------
# Generates image pairs for training a classification model
# - Reads metadata.jsonl
# - Creates:
#     - Positive pairs: same shape_id, different views
#     - Negative pairs: different shape_ids
# - Saves output as pairs.jsonl/csv with:
#     - img1, img2, label (1 for same object, 0 for different)

import json
import random
from collections import defaultdict


class PairGen:
    def __init__(self, input_file, output_file, neg_samples_per_pos):
        self.INPUT_FILE = input_file
        self.OUTPUT_JSONL = output_file
        self.NEGATIVE_SAMPLES_PER_POSITIVE = neg_samples_per_pos

    def read_metadata(self, file_path):
        items = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                items.append(item)
        return items

    def group_by_shape_id(self, items):
        shape_dict = defaultdict(list)
        for item in items:
            shape_dict[item['shape_id']].append(item)
        return shape_dict

    def generate_positive_pairs(self, shape_dict):
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
        with open(output_file, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')

    def run(self):
        print(f"Reading metadata from {self.INPUT_FILE}...")
        items = self.read_metadata(self.INPUT_FILE)
        print(f"Found {len(items)} metadata entries.")

        shape_dict = self.group_by_shape_id(items)
        positives = self.generate_positive_pairs(shape_dict)
        print(f"Generated {len(positives)} positive pairs.")

        negatives = self.generate_negative_pairs(items, positives, self.NEGATIVE_SAMPLES_PER_POSITIVE)
        print(f"Generated {len(negatives)} negative pairs.")
        all_pairs = positives + negatives
        random.shuffle(all_pairs)

        print(f"Saving {len(all_pairs)} pairs to {self.OUTPUT_JSONL}...")
        self.save_jsonl(all_pairs, self.OUTPUT_JSONL)
        print("Done.")