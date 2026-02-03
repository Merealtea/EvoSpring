import tensorflow as tf
import functools
import json
import os
import numpy as np
import h5py
import hydra
from omegaconf import DictConfig
import tensorflow.compat.v1 as tf1

# Enable eager execution
tf1.enable_eager_execution()


TF_DATASET_PATH = "/mnt/pool1/cxy/phystwin-v2/cylinder_flow/cylinder_flow"
SAVE_ROOT = "/mnt/pool1/cxy/phystwin-v2/meshdata"
DATA_KEYS = ["node_type", "cells", "mesh_pos", "velocity","pressure"]

def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds, meta  # Modified to return meta as well for later use

def convert_tfrecord_to_h5(tf_dataset_path, save_root, data_keys):
    """Convert TFRecord dataset to H5 files and calculate stats."""
    os.makedirs(save_root, exist_ok=True)
    
    # 1. Initialize stats accumulators
    # We use running sum and running squared sum to compute mean/std without loading all data to RAM
    # Structure: { key: { 'sum': np.array, 'sq_sum': np.array, 'count': int } }
    train_stats = {} 
    original_meta = None

    for split in ["train", "test", "valid"]:
        ds, meta = load_dataset(tf_dataset_path, split)
        
        # Keep a copy of meta to save later
        if original_meta is None:
            original_meta = meta

        split_dir = os.path.join(save_root, split)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"Processing split: {split}...")

        for index, d in enumerate(ds):
            try:
                # Convert tensors to numpy
                data = {key: d[key].numpy() for key in data_keys}
                
                # --- Statistical Accumulation Logic (Only for Train split) ---
                if split == "train":
                    for key, val in data.items():
                        # Only normalize floating point data (exclude cells/node_type)
                        if np.issubdtype(val.dtype, np.floating):
                            # Flatten data to shape (-1, feature_dim) to aggregate over time and nodes
                            # e.g., (Time, Nodes, 2) -> (Time*Nodes, 2)
                            flat_val = val.reshape(-1, val.shape[-1])
                            
                            if key not in train_stats:
                                train_stats[key] = {
                                    'sum': np.zeros(val.shape[-1]), 
                                    'sq_sum': np.zeros(val.shape[-1]), 
                                    'count': 0
                                }
                            
                            # Accumulate sum and squared sum
                            train_stats[key]['sum'] += np.sum(flat_val, axis=0)
                            train_stats[key]['sq_sum'] += np.sum(flat_val ** 2, axis=0)
                            train_stats[key]['count'] += flat_val.shape[0]

                # --- Save H5 File ---
                save_path = os.path.join(split_dir, f"{index}.h5")
                with h5py.File(save_path, "w") as f:
                    for key, value in data.items():
                        f.create_dataset(key, data=value)
                
                if index % 100 == 0:
                    print(f"[{split}] Processed index: {index}")

            except Exception as e:
                print(f"Skipped error in index: {index}, error: {e}")
                continue

    # 2. Compute Final Mean and Std
    print("Computing normalization statistics...")
    normalization_info = {}
    
    for key, stats in train_stats.items():
        total_count = stats['count']
        if total_count > 0:
            # E[X]
            mean = stats['sum'] / total_count
            # Var(X) = E[X^2] - (E[X])^2
            variance = (stats['sq_sum'] / total_count) - (mean ** 2)
            # Clip variance to avoid negative values due to float precision errors
            variance = np.maximum(variance, 0)
            std = np.sqrt(variance)

            # Store as list for JSON serialization
            normalization_info[key] = {
                "mean": mean.tolist(),
                "std": std.tolist()
            }
    
    # 3. Update Meta and Save
    if original_meta:
        original_meta["normalization_info"] = normalization_info
        new_meta_path = os.path.join(save_root, "meta.json")
        with open(new_meta_path, "w") as fp:
            json.dump(original_meta, fp, indent=4)
        print(f"New meta.json with normalization_info saved to {new_meta_path}")

def main():
    tf_dataset_path = TF_DATASET_PATH
    save_root = SAVE_ROOT
    data_keys = DATA_KEYS

    convert_tfrecord_to_h5(tf_dataset_path, save_root, data_keys)


if __name__ == "__main__":
    main()