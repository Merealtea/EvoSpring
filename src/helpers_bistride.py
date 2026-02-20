import numpy as np
import scipy
import scipy
# 替换原来的 from sparse_dot_mkl import dot_product_mkl
try:
    from sparse_dot_mkl import dot_product_mkl
except ImportError:
    # 定义一个兼容函数，使用 Scipy 替代 MKL
    def dot_product_mkl(A, B):
        # Scipy 的 @ 运算符或 .dot() 在处理稀疏矩阵时非常高效
        return A @ B
from enum import Enum
from helpers_convert import _flat_edge_to_adj_list, _flat_edge_to_adj_mat, _adj_mat_to_flat_edge
from helpers_BFS import _find_clusters, _BFS_dist, _BFS_dist_all
from helpers_contact import contact_edge, contact_edge_no_self


class SeedingHeuristic(Enum):
    MinAve = 1
    NearCenter = 2


_INF = 1 + 1e10


def _remove_invalid_connection(flat_list, node_type, invalid_markers):
    node_type = np.copy(node_type)
    node_type = node_type.astype(int)
    E = flat_list.shape[-1]
    new_flat_list = []
    for i in range(E - 1, -1, -1):
        s, r = flat_list[0, i], flat_list[1, i]
        if (node_type[s] in invalid_markers) and (node_type[r] in invalid_markers):
            pass
        else:
            new_flat_list.append([s, r])
    new_flat_list_np = np.array(new_flat_list).transpose()
    return new_flat_list_np


def _rnd_seed(clusters):
    seeds = []
    for c in clusters:
        rnd_node = np.random.choice(c)
        seeds.append(rnd_node)

    return seeds

def _min_ave_seed(adj_list, clusters):
    seeds = []
    dist = _BFS_dist_all(adj_list, len(adj_list))
    for c in clusters:
        d_c = dist[c]
        d_c = d_c[:, c]
        d_sum = np.sum(d_c, axis=1)
        min_ave_depth_node = c[np.argmin(d_sum)]
        seeds.append(min_ave_depth_node)

    return seeds


def _nearest_center_seed(adj_list, clusters, pos_mesh):
    seeds = []
    for c in clusters:
        center = np.mean(pos_mesh[c], axis=0)
        dd = pos_mesh[c] - center[None, :]
        normd = np.linalg.norm(dd, 2, axis=-1)
        thresh_d = np.min(normd) * 1.2
        tmp = np.where(normd < thresh_d)[0].tolist()
        try_node = [c[i] for i in tmp]
        # print(try_node)
        min_node = try_node[0]
        d_min, _ = _BFS_dist(adj_list, len(adj_list), min_node)
        min_d_sum = np.sum(d_min)
        for i in range(1, len(try_node)):
            trial = try_node[i]
            d_trial, _ = _BFS_dist(adj_list, len(adj_list), trial)
            d_trial_sum = np.sum(d_trial)
            if d_trial_sum < min_d_sum:
                min_node = trial
                min_d_sum = d_trial_sum
        seeds.append(min_node)

    return seeds


def pool_edge(g, idx, num_nodes):
    # g in scipy sparse mat
    g = _adj_mat_to_flat_edge(g)  # now flat edge list
    # idx is list, the node kept for subgraph
    idx = np.array(idx, dtype=np.longlong)
    idx_new_valid = np.arange(len(idx)).astype(np.longlong)
    idx_new_all = -1 * np.ones(num_nodes).astype(np.longlong)
    idx_new_all[idx] = idx_new_valid

    # g is the edge info
    new_g = -1 * np.ones_like(g).astype(np.longlong)
    new_g[0] = idx_new_all[g[0]]
    new_g[1] = idx_new_all[g[1]]
    both_valid = np.logical_and(new_g[0] >= 0, new_g[1] >= 0)
    e_idx = np.where(both_valid)[0]
    new_g = new_g[:, e_idx]
    return new_g


def bstride_selection(flat_edge, seed_heuristic, pos_mesh=None, n=None):
    combined_idx_kept = set()
    adj_list = _flat_edge_to_adj_list(flat_edge, n=n)
    adj_mat = _flat_edge_to_adj_mat(flat_edge, n=n)
    # adj mat enhance the diag
    adj_mat.setdiag(1)
    # 0. compute clusters, each of which should be deivded independantly
    clusters = _find_clusters(adj_list)
    # 1. seeding: by BFS_all for small graphs, or by seed_heuristic for larger graphs

    if seed_heuristic == SeedingHeuristic.NearCenter:
        seeds = _nearest_center_seed(adj_list, clusters, pos_mesh)
    else:
        # print('rnd_seed')
        # exit()
        # seeds = _rnd_seed(clusters) #_min_ave_seed(adj_list, clusters)
        seeds = _min_ave_seed(adj_list, clusters)

    for seed, c in zip(seeds, clusters):
        n_c = len(c)
        odd = set()
        even = set()
        index_kept = set()
        dist_from_cental_node, _ = _BFS_dist(adj_list, len(adj_list), seed)
        for i in range(len(dist_from_cental_node)):
            if dist_from_cental_node[i] % 2 == 0 and dist_from_cental_node[i] != _INF:
                even.add(i)
            elif dist_from_cental_node[i] % 2 == 1 and dist_from_cental_node[i] != _INF:
                odd.add(i)

        # 4. enforce n//2 candidates
        if len(even) <= len(odd) or len(odd) == 0:
            index_kept = even
            index_rmvd = odd
            delta = len(index_rmvd) - len(index_kept)
        else:
            index_kept = odd
            index_rmvd = even
            delta = len(index_rmvd) - len(index_kept)

        if delta > 0:
            # sort the dist of idx rmvd
            # cal stride based on delta nodes to select
            # generate strided idx from rmvd idx
            # union
            index_rmvd = list(index_rmvd)
            dist_id_rmvd = np.array(dist_from_cental_node)[index_rmvd]
            sort_index = np.argsort(dist_id_rmvd)
            stride = len(index_rmvd) // delta + 1
            delta_idx = sort_index[0::stride]
            delta_idx = set([index_rmvd[i] for i in delta_idx])
            index_kept = index_kept.union(delta_idx)

        combined_idx_kept = combined_idx_kept.union(index_kept)

    combined_idx_kept = list(combined_idx_kept)
    adj_mat = adj_mat.tocsr().astype(float)

    adj_mat = adj_mat @ adj_mat
    # adj_mat = dot_product_mkl(adj_mat, adj_mat) # BUG: cause indices overflow for large graph, need to check if the edge exists before dot product

    adj_mat.setdiag(0)

    adj_mat = pool_edge(adj_mat, combined_idx_kept, n)
    return combined_idx_kept, adj_mat

def multi_layer_contact_edge(m_gs, multi_layer_idx, pos, contact_radius, self_contact=True, init_contact_g=None):
    # mgs init_contact_g always in flat edge
    if not isinstance(init_contact_g, np.ndarray):
        # init contact g using contact radius
        if self_contact:
            init_contact_g = contact_edge(pos, m_gs[0], contact_radius)
        else:
            init_contact_g = contact_edge_no_self(pos, m_gs[0], contact_radius)

    multi_layer_contact_g = [init_contact_g]
    g_contact = _flat_edge_to_adj_mat(init_contact_g, n=pos.shape[-2])
    tempmgs = []
    for d, g in enumerate(m_gs):
        if d == 0:
            tempmgs.append(_flat_edge_to_adj_mat(g, n=pos.shape[-2]))
        else:
            tempmgs.append(_flat_edge_to_adj_mat(g, n=len(multi_layer_idx[d - 1])))
    m_gs = tempmgs
    for g in m_gs:
        g.setdiag(1)
    for d in range(len(multi_layer_idx)):
        n = pos.shape[-2] if d == 0 else len(multi_layer_idx[d - 1])
        g = m_gs[d]
        idx = multi_layer_idx[d]

        g_contact.setdiag(0)
        g = g.tocsr().astype(float)
        g_contact = g_contact.tocsr().astype(float)
        g_contact = dot_product_mkl(g_contact, g)
        g_contact = dot_product_mkl(g, g_contact)
        g_contact = g_contact.astype(bool).astype(float)
        g_contact.setdiag(0)
        g_contact = pool_edge(g_contact, idx, n)
        multi_layer_contact_g.append(g_contact)
        g_contact = _flat_edge_to_adj_mat(g_contact, n=len(idx))

    return multi_layer_contact_g


def _find_edge_parents(prev_edges, curr_edges, prev_adj_mat, node_mapping):
    """
    Find parent edges for each edge in the current layer.

    Args:
        prev_edges: edges from previous layer (2 x E_prev)
        curr_edges: edges from current layer (2 x E_curr)
        prev_adj_mat: adjacency matrix from previous layer (before pooling)
        node_mapping: mapping from old node indices to new node indices

    Returns:
        edge_parents: list of tuples, where each tuple contains:
            - If direct edge: (parent_edge_idx, -1)
            - If merged edge: (parent_edge_idx_1, parent_edge_idx_2)
    """
    edge_parents = []

    # Convert to CSR format for efficient indexing
    prev_adj_mat = prev_adj_mat.tocsr()

    # Create a mapping from edge (as tuple) to edge index in previous layer
    prev_edge_dict = {}
    for e_idx in range(prev_edges.shape[1]):
        edge = (prev_edges[0, e_idx], prev_edges[1, e_idx])
        prev_edge_dict[edge] = e_idx
        # Also add reverse edge since graph is undirected
        prev_edge_dict[(edge[1], edge[0])] = e_idx

    # Create reverse node mapping (new -> old)
    reverse_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(node_mapping)}
    print("finish prev_edge graph construction")

    for e_idx in range(curr_edges.shape[1]):
        src_new, dst_new = curr_edges[0, e_idx], curr_edges[1, e_idx]
        src_old, dst_old = reverse_mapping[src_new], reverse_mapping[dst_new]

        # Check if this edge existed directly in previous layer
        edge_key = (src_old, dst_old)

        if edge_key in prev_edge_dict:
            # Direct edge from previous layer
            edge_parents.append((prev_edge_dict[edge_key], -1))
        else:
            # Merged edge - find intermediate nodes
            # This edge was created by A^2, so there must be intermediate node(s)
            # Only iterate through neighbors of src_old for efficiency
            src_neighbors = prev_adj_mat._getrow(src_old).indices
            intermediate_node = None
            for k in src_neighbors:
                if prev_adj_mat[k, dst_old] > 0:
                    intermediate_node = k
                    break

            if intermediate_node is not None:
                # Use the first intermediate node to identify the two parent edges
                k = intermediate_node
                edge1_key = (src_old, intermediate_node)
                # edge1_key_rev = (k, src_old)
                edge2_key = (intermediate_node, dst_old)
                # edge2_key_rev = (dst_old, k)

                parent1 = prev_edge_dict.get(edge1_key, -1)
                parent2 = prev_edge_dict.get(edge2_key, -1)
                edge_parents.append((parent1, parent2))
            else:
                # Shouldn't happen, but handle gracefully
                edge_parents.append((-1, -1))

    return edge_parents


def generate_multi_layer_stride(flat_edge, num_l, seed_heuristic, n, pos_mesh=None):
    m_gs = [flat_edge]
    m_ids = []
    m_edge_parents = []  # New: track edge parent relationships
    g = flat_edge

    for l in range(num_l):
        n_l = n if l == 0 else len(index_to_keep)  # n_l is the number of nodes in the current layer

        # Get edge mapping information
        result = bstride_selection(g, seed_heuristic=seed_heuristic, pos_mesh=pos_mesh, n=n_l)

        index_to_keep, g_new = result

        # Find parent edges for each edge in the new layer
        prev_adj_mat = _flat_edge_to_adj_mat(g, n=n_l)        
        edge_parents = _find_edge_parents(g, g_new, prev_adj_mat, index_to_keep)


        pos_mesh = pos_mesh[index_to_keep]
        m_gs.append(g_new)
        m_ids.append(index_to_keep)
        m_edge_parents.append(edge_parents)

        g = g_new

    return m_gs, m_ids, m_edge_parents

