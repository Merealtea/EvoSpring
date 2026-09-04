#!/usr/bin/env python3
"""
Extract node & edge statistics from all cases' global_best_mech_info.pth files.
Output: visualization/connection_strength/all_cases_node_stats.csv
"""

import os
import sys
import glob
import csv
import numpy as np
import torch

BASE_RES_DIR = os.path.join(os.path.dirname(__file__),
                            "res/End2EndReductionLearnable4Downsample")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__),
                          "visualization/connection_strength/all_cases_node_stats.csv")

SKIP_DIRS = {'extract_stage_times.py', 'stage_times.csv'}

NODE_TYPE_NAMES = {0: 'object', 1: 'surface', 2: 'interior', 3: 'controller'}

YOUNG_THRESHOLD = 10.0
MAX_EDGE_DIST = 0.05


def tn(x):
    """Safe tensor -> numpy: handles requires_grad via detach."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def discover_cases(base_dir):
    """Find all cases with valid mech_info pth files."""
    cases = []
    for entry in sorted(os.listdir(base_dir)):
        case_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(case_dir) or entry in SKIP_DIRS:
            continue
        pth_files = glob.glob(os.path.join(case_dir, '*/spring_mech_info/global_best_mech_info.pth'))
        if pth_files:
            cases.append((entry, pth_files[0]))
    return cases


def extract_case_stats(case_name, pth_path):
    """Extract statistics for all levels of a case. Returns list of dict rows."""
    data = torch.load(pth_path, map_location='cpu')
    mech_list = data['mech'] if isinstance(data, dict) and 'mech' in data else [data]

    rows = []
    for level_idx, level in enumerate(mech_list):
        verts = tn(level['vertices'])
        edges = tn(level['edges'])
        log_Y = tn(level['log_spring_Y'])
        node_types = tn(level['node_type']).flatten()
        masses = tn(level['masses']).flatten()
        gt_verts = tn(level['gt_vertices'])

        # Node counts
        n_total = len(node_types)
        n_obj = int(np.sum(node_types == 0))
        n_surf = int(np.sum(node_types == 1))
        n_int = int(np.sum(node_types == 2))
        n_ctrl = int(np.sum(node_types == 3))
        n_object_points = n_obj + n_surf + n_int  # all non-controller

        # Edge counts
        n_edges_directed = edges.shape[1]

        # Undirected edges (dedup)
        src, dst = edges[0], edges[1]
        pairs = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=0).T
        _, idx = np.unique(pairs, axis=0, return_index=True)
        n_edges_undirected = len(idx)

        # Young's modulus
        Y = np.exp(log_Y)
        Y_undir = Y[np.sort(idx)]

        # After Y >= 10 filter
        Y_filt = Y_undir[Y_undir >= YOUNG_THRESHOLD]
        n_edges_y_filtered = len(Y_filt)

        # After distance filter (on Y-filtered edges)
        undir_edges = edges[:, np.sort(idx)]
        filt_edges = undir_edges[:, Y_undir >= YOUNG_THRESHOLD]
        if filt_edges.shape[1] > 0:
            s, d = filt_edges[0].astype(np.int64), filt_edges[1].astype(np.int64)
            dists = np.linalg.norm(verts[s] - verts[d], axis=1)
            Y_filt = Y_filt[dists <= MAX_EDGE_DIST]
            n_edges_final = int(np.sum(dists <= MAX_EDGE_DIST))
        else:
            n_edges_final = 0

        # Young's modulus stats (after all filters)
        if len(Y_filt) > 0:
            y_min, y_max = Y_filt.min(), Y_filt.max()
            y_mean, y_median, y_std = Y_filt.mean(), np.median(Y_filt), Y_filt.std()
        else:
            y_min = y_max = y_mean = y_median = y_std = 0

        # Mass stats
        mass_min, mass_max = masses.min(), masses.max()
        mass_mean, mass_median, mass_std = masses.mean(), np.median(masses), masses.std()

        # Controller mass stats
        ctrl_masses = masses[node_types == 3]
        if len(ctrl_masses) > 0:
            ctrl_mass_mean = ctrl_masses.mean()
            ctrl_mass_std = ctrl_masses.std()
        else:
            ctrl_mass_mean = ctrl_mass_std = 0

        # Object mass stats
        obj_masses = masses[node_types != 3]
        if len(obj_masses) > 0:
            obj_mass_mean = obj_masses.mean()
            obj_mass_std = obj_masses.std()
        else:
            obj_mass_mean = obj_mass_std = 0

        # Vertex bounds
        vx_min, vx_max = verts[:, 0].min(), verts[:, 0].max()
        vy_min, vy_max = verts[:, 1].min(), verts[:, 1].max()
        vz_min, vz_max = verts[:, 2].min(), verts[:, 2].max()

        # Bounding box size
        dx = vx_max - vx_min
        dy = vy_max - vy_min
        dz = vz_max - vz_min

        # GT data
        gt_frames = gt_verts.shape[0]
        gt_nodes = gt_verts.shape[1]

        # Has node_ids
        has_node_ids = 'node_ids' in level and level['node_ids'] is not None

        # Mechanical params
        drag_damping = float(tn(level['drag_damping']).item())
        dashpot_damping = float(tn(level['dashpot_damping']).item())
        collision_elas = float(tn(level['collision_elas']).item())
        collision_fric = float(tn(level['collision_fric']).item())
        collision_obj_elas = float(tn(level['collision_object_elas']).item())
        collision_obj_fric = float(tn(level['collision_object_fric']).item())

        row = {
            'case': case_name,
            'level': level_idx,
            # Node counts
            'n_total': n_total,
            'n_object': n_obj,
            'n_surface': n_surf,
            'n_interior': n_int,
            'n_controller': n_ctrl,
            'n_object_points': n_object_points,
            # Edge counts
            'n_edges_directed': n_edges_directed,
            'n_edges_undirected': n_edges_undirected,
            'n_edges_Y_filtered': n_edges_y_filtered,
            'n_edges_final': n_edges_final,
            'pct_edges_kept': round(100 * n_edges_final / max(n_edges_undirected, 1), 2),
            # Y stats
            'Y_min': round(y_min, 2),
            'Y_max': round(y_max, 2),
            'Y_mean': round(y_mean, 2),
            'Y_median': round(y_median, 2),
            'Y_std': round(y_std, 2),
            # Mass stats
            'mass_min': round(float(mass_min), 6),
            'mass_max': round(float(mass_max), 6),
            'mass_mean': round(float(mass_mean), 6),
            'mass_median': round(float(mass_median), 6),
            'mass_std': round(float(mass_std), 6),
            'ctrl_mass_mean': round(float(ctrl_mass_mean), 6),
            'obj_mass_mean': round(float(obj_mass_mean), 6),
            # Vertex bounds
            'v_x_min': round(float(vx_min), 4),
            'v_x_max': round(float(vx_max), 4),
            'v_y_min': round(float(vy_min), 4),
            'v_y_max': round(float(vy_max), 4),
            'v_z_min': round(float(vz_min), 4),
            'v_z_max': round(float(vz_max), 4),
            'bbox_dx': round(float(dx), 4),
            'bbox_dy': round(float(dy), 4),
            'bbox_dz': round(float(dz), 4),
            # GT data
            'gt_frames': gt_frames,
            'gt_nodes': gt_nodes,
            # Meta
            'has_node_ids': has_node_ids,
            # Mechanical params
            'drag_damping': drag_damping,
            'dashpot_damping': dashpot_damping,
            'collision_elas': collision_elas,
            'collision_fric': collision_fric,
            'collision_obj_elas': collision_obj_elas,
            'collision_obj_fric': collision_obj_fric,
        }
        rows.append(row)

    return rows


def main():
    cases = discover_cases(BASE_RES_DIR)
    print(f"Found {len(cases)} cases.")

    all_rows = []
    for case_name, pth_path in cases:
        print(f"  Processing {case_name}...")
        try:
            rows = extract_case_stats(case_name, pth_path)
            all_rows.extend(rows)
            print(f"    {len(rows)} levels extracted.")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = list(all_rows[0].keys()) if all_rows else []

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows ({len(cases)} cases × up to 4 levels) to:")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
