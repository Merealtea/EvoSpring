#!/usr/bin/env python3
"""
Node Connection Strength Visualization — Draws edges colored by Young's Modulus (Y).

- Edges with Y < 10 are skipped (negligible springs)
- Edges with node distance > 0.05m are skipped
- Three orthographic views: Front, Side, Top
- Object points (interior + exterior + surface) → one color
- Controller points → another color
- Edge color: dark (low Y) → bright (high Y), log-scaled

Usage:
    python connection_strength_visualization.py              # all cases
    python connection_strength_visualization.py CASE_NAME    # single case
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import torch

# ============================================================
# Constants
# ============================================================
NODE_TYPE_OBJECT = 0
NODE_TYPE_SURFACE = 1
NODE_TYPE_INTERIOR = 2
NODE_TYPE_CONTROLLER = 3

OBJECT_POINT_COLOR = '#5D8AA8'
CONTROLLER_POINT_COLOR = '#E25822'

YOUNG_MODULUS_THRESHOLD = 10.0
MAX_EDGE_DISTANCE = 0.05

OBJECT_POINT_SIZE = 2.0
CONTROLLER_POINT_SIZE = 35.0

EDGE_LINEWIDTH = 0.4
EDGE_ALPHA = 0.7

BASE_RES_DIR = os.path.join(os.path.dirname(__file__),
                            "res/End2EndReductionLearnable4Downsample")
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__),
                               "visualization/connection_strength")

# Non-case directories to skip
SKIP_DIRS = {'extract_stage_times.py', 'stage_times.csv'}

# View angles: (elev, azim)
VIEW_CONFIGS = [
    ('front', 0, 0,   'Front View'),
    ('side',  0, 90,  'Side View'),
    ('top',   90, 0,  'Top View'),
]


def discover_cases(base_dir):
    """Find all cases with valid mech_info pth files. Returns list of (case_name, pth_path)."""
    cases = []
    for entry in sorted(os.listdir(base_dir)):
        case_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(case_dir) or entry in SKIP_DIRS:
            continue
        # Find the pth file inside any timestamp subdirectory
        pth_files = glob.glob(os.path.join(case_dir, '*/spring_mech_info/global_best_mech_info.pth'))
        if pth_files:
            cases.append((entry, pth_files[0]))
        else:
            print(f"  SKIP {entry}: no pth file found")
    return cases


def load_mech_info(path):
    """Load multi-level mech_info from .pth file."""
    loaded = torch.load(path, map_location='cpu')
    if isinstance(loaded, dict) and 'mech' in loaded:
        return loaded['mech']
    return [loaded]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return np.asarray(x)


def deduplicate_edges(edges, spring_Y):
    """Convert directed edges [2, E] to undirected, keeping src < dst."""
    src, dst = edges[0], edges[1]
    pairs = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=0).T
    _, idx = np.unique(pairs, axis=0, return_index=True)
    idx = np.sort(idx)
    return edges[:, idx], spring_Y[idx]


def filter_edges(edges, spring_Y, threshold=YOUNG_MODULUS_THRESHOLD):
    mask = spring_Y >= threshold
    return edges[:, mask], spring_Y[mask]


def filter_edges_by_distance(edges, spring_Y, verts, max_dist=MAX_EDGE_DISTANCE):
    src, dst = edges[0].astype(np.int64), edges[1].astype(np.int64)
    dist = np.linalg.norm(verts[src] - verts[dst], axis=1)
    mask = dist <= max_dist
    return edges[:, mask], spring_Y[mask]


def build_edge_segments(verts, edges):
    src, dst = edges[0].astype(np.int64), edges[1].astype(np.int64)
    valid = (src < len(verts)) & (dst < len(verts))
    src, dst = src[valid], dst[valid]
    segments = np.stack([
        np.column_stack([verts[src, 0], verts[src, 1], verts[src, 2]]),
        np.column_stack([verts[dst, 0], verts[dst, 1], verts[dst, 2]]),
    ], axis=1)
    return segments, valid


def draw_three_view_figure(verts, segments, edge_colors, obj_mask, ctrl_mask,
                           norm, cmap, bounds, case_name, level_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'},
                             figsize=(24, 8))
    fig.suptitle(f'{case_name} — Level {level_idx} — Connection Strength',
                 fontsize=15, y=0.96)

    for ax_idx, (view_name, elev, azim, title) in enumerate(VIEW_CONFIGS):
        ax = axes[ax_idx]

        if len(segments) > 0:
            lc = Line3DCollection(segments, colors=edge_colors,
                                  linewidths=EDGE_LINEWIDTH, alpha=EDGE_ALPHA)
            ax.add_collection3d(lc)

        if np.sum(obj_mask) > 0:
            ax.scatter(verts[obj_mask, 0], verts[obj_mask, 1], verts[obj_mask, 2],
                       c=OBJECT_POINT_COLOR, s=OBJECT_POINT_SIZE, alpha=0.5,
                       label=f'Object pts ({np.sum(obj_mask)})', rasterized=True)

        if np.sum(ctrl_mask) > 0:
            ax.scatter(verts[ctrl_mask, 0], verts[ctrl_mask, 1], verts[ctrl_mask, 2],
                       c=CONTROLLER_POINT_COLOR, s=CONTROLLER_POINT_SIZE, alpha=1.0,
                       edgecolors='#8B0000', linewidth=1.2,
                       label=f'Controller pts ({np.sum(ctrl_mask)})', rasterized=True)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=13, pad=8)
        ax.legend(loc='upper right', fontsize=8, markerscale=0.8)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.915, 0.18, 0.012, 0.60])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Young's Modulus (Y)", fontsize=11)
    cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(left=0.04, right=0.90, top=0.90, bottom=0.06, wspace=0.15)

    out_path = os.path.join(output_dir, f'level_{level_idx}_three_views.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def draw_iso_overview(verts, segments, edge_colors, obj_mask, ctrl_mask,
                      norm, cmap, bounds, spring_Y, case_name, level_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    if len(segments) > 0:
        lc = Line3DCollection(segments, colors=edge_colors,
                              linewidths=EDGE_LINEWIDTH, alpha=EDGE_ALPHA)
        ax.add_collection3d(lc)

    if np.sum(obj_mask) > 0:
        ax.scatter(verts[obj_mask, 0], verts[obj_mask, 1], verts[obj_mask, 2],
                   c=OBJECT_POINT_COLOR, s=OBJECT_POINT_SIZE, alpha=0.5,
                   label=f'Object pts ({np.sum(obj_mask)})', rasterized=True)

    if np.sum(ctrl_mask) > 0:
        ax.scatter(verts[ctrl_mask, 0], verts[ctrl_mask, 1], verts[ctrl_mask, 2],
                   c=CONTROLLER_POINT_COLOR, s=CONTROLLER_POINT_SIZE * 1.5, alpha=1.0,
                   edgecolors='#8B0000', linewidth=1.5,
                   label=f'Controller pts ({np.sum(ctrl_mask)})', rasterized=True)

    ax.view_init(elev=25, azim=45)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{case_name} — Level {level_idx} — Overview', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.08, shrink=0.5)
    cbar.set_label("Young's Modulus (Y)", fontsize=11)

    y_min_val = max(spring_Y.min(), YOUNG_MODULUS_THRESHOLD)
    y_max_val = spring_Y.max()
    stats = (
        f"Case: {case_name}  Level {level_idx}\n"
        f"  Total nodes:          {verts.shape[0]}\n"
        f"  Object points:        {np.sum(obj_mask)}\n"
        f"  Controller points:    {np.sum(ctrl_mask)}\n"
        f"  Edges shown:          {len(segments)}\n"
        f"  Y range:              [{y_min_val:.1f}, {y_max_val:.1f}]\n"
        f"  Y median:             {np.median(spring_Y):.1f}"
    )
    fig.text(0.02, 0.02, stats, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    out_path = os.path.join(output_dir, f'level_{level_idx}_overview.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def print_stats(case_name, level_idx, verts, node_types, orig_directed,
                undirected_count, y_filtered_count, final_count, spring_Y):
    types = to_numpy(node_types).flatten()
    print(f"  L{level_idx}: {verts.shape[0]} nodes "
          f"(obj={np.sum(types==0)}, surf={np.sum(types==1)}, "
          f"int={np.sum(types==2)}, ctrl={np.sum(types==3)}) | "
          f"edges: {orig_directed}->{undirected_count}->{y_filtered_count}(Y>={YOUNG_MODULUS_THRESHOLD})->{final_count}(d<={MAX_EDGE_DISTANCE}m) "
          f"({100*final_count/max(undirected_count,1):.0f}%) | "
          f"Y: [{spring_Y.min():.0f}, {spring_Y.max():.0f}] med={np.median(spring_Y):.0f}")


def process_level(level_data, level_idx, case_name, output_dir):
    verts = to_numpy(level_data['vertices'])
    edges_raw = to_numpy(level_data['edges'])
    log_Y = to_numpy(level_data['log_spring_Y'])
    node_types = to_numpy(level_data['node_type'])

    spring_Y_raw = np.exp(log_Y)
    orig_directed = edges_raw.shape[1]

    undir_edges, undir_Y = deduplicate_edges(edges_raw, spring_Y_raw)
    undirected_count = undir_edges.shape[1]

    filt_edges, filt_Y = filter_edges(undir_edges, undir_Y, YOUNG_MODULUS_THRESHOLD)
    y_filtered_count = filt_edges.shape[1]

    filt_edges, filt_Y = filter_edges_by_distance(filt_edges, filt_Y, verts, MAX_EDGE_DISTANCE)
    filtered_count = filt_edges.shape[1]

    print_stats(case_name, level_idx, verts, node_types, orig_directed,
                undirected_count, y_filtered_count, filtered_count, filt_Y)

    if filtered_count == 0:
        return

    types_flat = node_types.flatten()
    obj_mask = (types_flat != NODE_TYPE_CONTROLLER)
    ctrl_mask = (types_flat == NODE_TYPE_CONTROLLER)

    segments, valid_mask = build_edge_segments(verts, filt_edges)
    valid_spring_Y = filt_Y[valid_mask]

    if len(segments) == 0:
        return

    y_min_val = max(valid_spring_Y.min(), YOUNG_MODULUS_THRESHOLD)
    y_max_val = valid_spring_Y.max()
    if y_max_val <= y_min_val:
        y_max_val = y_min_val + 1.0

    norm = LogNorm(vmin=y_min_val, vmax=y_max_val)
    cmap = plt.cm.viridis
    edge_colors = cmap(norm(valid_spring_Y))

    margin = 0.15
    bounds = (verts[:, 0].min() - margin, verts[:, 0].max() + margin,
              verts[:, 1].min() - margin, verts[:, 1].max() + margin,
              verts[:, 2].min() - margin, verts[:, 2].max() + margin)

    draw_three_view_figure(verts, segments, edge_colors, obj_mask, ctrl_mask,
                           norm, cmap, bounds, case_name, level_idx, output_dir)
    draw_iso_overview(verts, segments, edge_colors, obj_mask, ctrl_mask,
                      norm, cmap, bounds, valid_spring_Y, case_name, level_idx, output_dir)


def process_case(case_name, pth_path):
    print(f"\n{'='*60}")
    print(f"[{case_name}]")
    print(f"{'='*60}")

    try:
        mech_list = load_mech_info(pth_path)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return

    output_dir = os.path.join(BASE_OUTPUT_DIR, case_name)
    print(f"  {len(mech_list)} levels -> {output_dir}")

    for level_idx, level_data in enumerate(mech_list):
        try:
            process_level(level_data, level_idx, case_name, output_dir)
        except Exception as e:
            print(f"  L{level_idx} ERROR: {e}")
            import traceback
            traceback.print_exc()


def main():
    print(f"\n{'='*60}")
    print(f"Connection Strength Visualization — Batch Mode")
    print(f"Y threshold: >= {YOUNG_MODULUS_THRESHOLD}  |  "
          f"Max edge distance: <= {MAX_EDGE_DISTANCE}m")
    print(f"{'='*60}")

    # Discover cases
    all_cases = discover_cases(BASE_RES_DIR)
    print(f"\nFound {len(all_cases)} cases with mech_info data.")

    # Filter by command-line arg if provided
    if len(sys.argv) > 1:
        target = sys.argv[1]
        all_cases = [(n, p) for n, p in all_cases if target in n]
        if not all_cases:
            print(f"No case matching '{target}' found.")
            return
        print(f"Filtered to {len(all_cases)} matching case(s).")

    # Process each case
    for i, (case_name, pth_path) in enumerate(all_cases):
        print(f"\n[{i+1}/{len(all_cases)}]", end="")
        process_case(case_name, pth_path)

    print(f"\n{'='*60}")
    print(f"Done. {len(all_cases)} case(s) processed.")
    print(f"Output base: {BASE_OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
