import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import solve_continuous_lyapunov, cholesky
from scipy.cluster.hierarchy import linkage, fcluster
import logging
from time import time
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering

def build_physical_matrices_sparse(m_vector, m_gs, Y_vector, d_dashpot, d_drag):
    """
    构建 稀疏格式(Sparse) 的 M, D, L 矩阵，内存占用接近零。
    """
    n = len(m_vector)
    
    # 1. 稀疏质量矩阵 M
    M_sparse = sp.diags(m_vector, format='csr')
    
    u = m_gs[0]
    v = m_gs[1]
    valid_mask = (u != v)
    try:
        u_v, v_v, Y_v = u[valid_mask], v[valid_mask], Y_vector[valid_mask]
    except:
        import pdb; pdb.set_trace()
    # 2. 稀疏拉普拉斯矩阵 L
    rows = np.concatenate([u_v, v_v])
    cols = np.concatenate([v_v, u_v])
    vals = np.concatenate([-Y_v, -Y_v])
    
    L_sparse = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # 计算对角线：每行的非对角线元素之和的相反数
    diag_L = -np.asarray(L_sparse.sum(axis=1)).flatten()
    L_sparse.setdiag(diag_L + L_sparse.diagonal())
    
    # 3. 稀疏阻尼矩阵 D
    if d_dashpot > 0 and len(u_v) > 0:
        dash_vals = np.full(len(u_v) * 2, -d_dashpot, dtype=np.float64)
        D_sparse = sp.coo_matrix((dash_vals, (rows, cols)), shape=(n, n)).tocsr()
        diag_D = -np.asarray(D_sparse.sum(axis=1)).flatten()
        D_sparse.setdiag(diag_D + d_drag)
    else:
        D_sparse = sp.diags(np.full(n, d_drag), format='csr')

    return M_sparse, D_sparse, L_sparse


class FastAdaptiveNetworkReducer:
    def __init__(self, distance_threshold='auto', num_modes=100):
        """
        :param distance_threshold: 聚类阈值
        :param num_modes: 模态截断保留的阶数
        """
        self.distance_threshold = distance_threshold
        self.num_modes = num_modes

    def reduce(self, m_vector, m_gs, spring_Y, d_dashpot, d_drag, node_type):
        st_global = time()    
        n = len(m_vector)
        
        # 1. 识别节点角色
        ctrl_indices = np.where(node_type == 3)[0]
        normal_indices = np.where(node_type != 3)[0]
        
        n_ctrl = len(ctrl_indices)
        n_normal = len(normal_indices)
        
        if n_normal <= 1:
            logging.warning("No normal nodes to reduce or system too small.")
            return np.eye(n), None, None, None

        # 2. 构建物理矩阵
        M_sp, D_sp, L_sp = build_physical_matrices_sparse(m_vector, 
                                                          m_gs, 
                                                          spring_Y, 
                                                          d_dashpot, 
                                                          d_drag)

        # =================================================================
        # 3. 提取受力节点 (公共逻辑提前)
        # =================================================================
        u, v = m_gs[0], m_gs[1]
        connected_v = v[np.isin(u, ctrl_indices)]
        connected_u = u[np.isin(v, ctrl_indices)]
        target_nodes = np.unique(np.concatenate([connected_u, connected_v]))
        target_nodes = target_nodes[node_type[target_nodes] != 3]
        
        m_inputs = len(target_nodes) if len(target_nodes) > 0 else 1
        if len(target_nodes) == 0: target_nodes = np.array([0])

        # =================================================================
        # 4. 判断并计算特征嵌入 Z (全空间 Lyapunov vs 降维投影 Lyapunov)
        # =================================================================
        if n < 2 * self.num_modes:
            # --- 不进行模态分解：直接在完整物理空间求解 Lyapunov ---
            print(f"Node count ({n}) < 2*num_modes. Skipping eigsh, solving full-space Lyapunov.")
            
            # 构造未降维的系统矩阵 A_full (2n x 2n)
            M_d = M_sp.diagonal()
            M_inv = sp.diags(1.0 / M_d, format='csr')
            Minv_L = (M_inv @ L_sp).toarray()
            Minv_D = (M_inv @ D_sp).toarray()
            
            A_full = np.block([
                [np.zeros((n, n)), np.eye(n)],
                [-Minv_L, -Minv_D]
            ])
            
            # 构造未降维的输入矩阵 B_full_eff
            ones_n = np.ones((n, 1))
            sigma_D = (ones_n.T @ D_sp @ ones_n)[0, 0]
            ones_T_F = np.ones((1, m_inputs))
            
            F_full = np.zeros((n, m_inputs))
            for i, idx in enumerate(target_nodes):
                F_full[idx, i] = 1.0
                
            B_full_eff = np.block([
                [-(1.0 / sigma_D) * ones_n @ ones_T_F],
                [F_full]
            ])
            
            # 求解全维 Lyapunov 方程
            try:
                P_tilde = solve_continuous_lyapunov(A_full - 1e-5 * np.eye(2 * n), -B_full_eff @ B_full_eff.T)
            except Exception as e:
                logging.error(f"Full-space Lyapunov failed: {e}")
                import pdb; pdb.set_trace()
                
            # 全维 Gramian 修正与嵌入空间计算
            nu_full_T = np.block([[ones_n.T @ D_sp, ones_n.T @ M_sp]])
            beta_a = - (nu_full_T @ P_tilde @ nu_full_T.T)[0, 0] / (sigma_D ** 2)
            Pi_full = np.block([
                [ones_n @ ones_n.T, np.zeros((n, n))],
                [np.zeros((n, n)), np.zeros((n, n))]
            ])
            P_gramian = P_tilde + beta_a * Pi_full
            
            P11 = P_gramian[:n, :n]
            w, v_e = np.linalg.eigh(P11)
            w[w < 0] = 0
            C = v_e @ np.diag(np.sqrt(w))
            Z = C  # 全空间下等价于 Phi=I_n，所以 Z = I * C = C

        else:
            # --- 原有逻辑：进行模态分解，在低维空间求解 Lyapunov ---
            k_modes = min(self.num_modes, n - 2) 
            eigenvalues, Phi = eigsh(L_sp, k=k_modes, M=M_sp, sigma=-1e-4, which='LM')
            
            for i in range(k_modes):
                norm = np.sqrt(Phi[:, i].T @ M_sp @ Phi[:, i])
                if norm > 1e-12: Phi[:, i] /= norm

            F_r = Phi[target_nodes, :].T  
            ones_n = np.ones((n, 1))
            sigma_D = (ones_n.T @ D_sp @ ones_n)[0, 0]
            ones_T_F = np.ones((1, m_inputs)) 

            B_r_eff = np.block([
                [-(1.0 / sigma_D) * (Phi.T @ ones_n) @ ones_T_F],
                [F_r]
            ])

            M_r = Phi.T @ M_sp @ Phi
            D_r = Phi.T @ D_sp @ Phi
            L_r = np.diag(eigenvalues)
            
            A_r = np.block([
                [np.zeros((k_modes, k_modes)), np.eye(k_modes)],
                [-np.linalg.inv(M_r) @ L_r, -np.linalg.inv(M_r) @ D_r]
            ])
            
            try:
                P_r_tilde = solve_continuous_lyapunov(A_r - 1e-5 * np.eye(2 * k_modes), -B_r_eff @ B_r_eff.T)
            except:
                import pdb; pdb.set_trace()
                
            nu_r_T = np.block([[ones_n.T @ D_sp @ Phi, ones_n.T @ M_sp @ Phi]])
            beta_a = - (nu_r_T @ P_r_tilde @ nu_r_T.T)[0, 0] / (sigma_D ** 2)
            Pi_r = np.block([
                [Phi.T @ ones_n @ ones_n.T @ Phi, np.zeros((k_modes, k_modes))],
                [np.zeros((k_modes, k_modes)), np.zeros((k_modes, k_modes))]
            ])
            P_r_gramian = P_r_tilde + beta_a * Pi_r

            P_r11 = P_r_gramian[:k_modes, :k_modes]
            w, v_e = np.linalg.eigh(P_r11)
            w[w < 0] = 0
            C = v_e @ np.diag(np.sqrt(w))
            Z = Phi @ C

        # =================================================================
        # 5. 限制性聚类逻辑：仅聚类 normal_indices
        # =================================================================
        Z_normal = Z[normal_indices, :]
        normal_types = node_type[normal_indices]
        n_nodes = len(normal_indices)
        
        global_to_local = np.full(n, -1, dtype=np.int32)
        global_to_local[normal_indices] = np.arange(n_nodes)
        
        u, v = m_gs[0], m_gs[1]
        u_local = global_to_local[u]
        v_local = global_to_local[v]
        
        valid_mask = (u_local >= 0) & (v_local >= 0) & (u_local != v_local)
        u_local = u_local[valid_mask]
        v_local = v_local[valid_mask]
        
        cluster_labels = self._constrained_hierarchical_clustering_vectorized(
            Z_normal, u_local, v_local, n_nodes, normal_types
        )

        # =================================================================
        # 6. 构建 N x N 投影矩阵 P
        # =================================================================
        P = np.zeros((n, n))
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            in_cluster_mask = (cluster_labels == label)
            cluster_global_indices = normal_indices[in_cluster_mask]
            
            Z_cluster = Z[cluster_global_indices]
            centroid = np.mean(Z_cluster, axis=0)
            
            dists = np.linalg.norm(Z_cluster - centroid, axis=1)
            j = cluster_global_indices[np.argmin(dists)]
            P[cluster_global_indices, j] = 1.0

        for ctrl_idx in ctrl_indices:
            P[ctrl_idx, ctrl_idx] = 1.0

        return P

    def _constrained_hierarchical_clustering_vectorized(self, Z_normal, edge_src, edge_dst, n_nodes, node_types=None):
        """
        执行约束层次聚类：只允许空间相连（有边连接）且 type 相同的节点合并
        完全使用矩阵运算实现，避免 for 循环
        
        Args:
            Z_normal: 普通节点的嵌入坐标 [n_normal, k]
            edge_src: 边的源节点局部索引 [n_edges]
            edge_dst: 边的目标节点局部索引 [n_edges]
            n_nodes: 普通节点数量
            node_types: 每个普通节点的 type 数组（用于约束只合并 type 相同的节点）
        
        Returns:
            cluster_labels: 每个节点的聚类标签
        """
        Z_src = Z_normal[edge_src]
        Z_dst = Z_normal[edge_dst]
        edge_distances = np.linalg.norm(Z_src - Z_dst, axis=1)

        dist_threshold = np.percentile(edge_distances, 10)

        # 3. 构造类型一致性约束 + 物理连接约束
        type_mask = (node_types[edge_src] == node_types[edge_dst])
        valid_u = edge_src[type_mask]
        valid_v = edge_dst[type_mask]
        
        # 构造连接矩阵（只有相连且同类型的点才能合并）
        connectivity = sp.coo_matrix(
            (np.ones(len(valid_u)), (valid_u, valid_v)),
            shape=(n_nodes, n_nodes)
        )

        # 4. 执行层次聚类
        # n_clusters=None 必须配合 distance_threshold 使用
        # linkage='average' (平均距离) 比之前的连通分量法能有效防止“一锅端”
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_threshold,
            connectivity=connectivity,
            linkage='average' 
        )
        
        labels = model.fit_predict(Z_normal)

        # --- 新增：打印普通节点的压缩情况 ---
        n_clusters = len(np.unique(labels))
        reduction_pct = (1 - n_clusters / n_nodes) * 100
        print(f"\n[Clustering Detail]")
        print(f" - Normal Nodes: {n_nodes} -> {n_clusters}")
        print(f" - Reduction Ratio: {reduction_pct:.2f}%")
        print(f" - Adaptive Threshold: {dist_threshold:.6f}")

        return labels


class RandomReducer:
    def __init__(self, ratio=0.5, seed=None):
        """
        随机Reducer，对普通节点进行随机合并
        
        :param ratio: 最终保留的普通节点比例，默认0.5
        :param seed: 随机种子，用于可重复性
        """
        self.ratio = ratio
        self.seed = seed

    def reduce(self, m_vector, m_gs, spring_Y, d_dashpot, d_drag, node_type):
        """
        执行随机降维
        
        Args:
            m_vector: 节点质量向量 [n]
            m_gs: 边的连接关系 [2, n_edges]
            spring_Y: 弹簧刚度系数 [n_edges]
            d_dashpot: 阻尼系数
            d_drag: 阻力系数
            node_type: 节点类型数组 [n]，type==3 为 controller
        
        Returns:
            P: 投影矩阵 [n, n]
            M_hat: 降维后的质量矩阵 [n, n]
            D_hat: 降维后的阻尼矩阵 [n, n]
            L_hat: 降维后的拉普拉斯矩阵 [n, n]
        """
        st_global = time()
        n = len(m_vector)
        
        # 1. 识别节点角色
        # node_type == 3 为 controller
        ctrl_indices = np.where(node_type == 3)[0]
        normal_indices = np.where(node_type != 3)[0]
        
        n_ctrl = len(ctrl_indices)
        n_normal = len(normal_indices)
        
        if n_normal <= 1:
            logging.warning("No normal nodes to reduce or system too small.")
            return np.eye(n), None, None, None

        # 2. 构建物理矩阵（使用提供的 sparse 构建函数）
        M_sp, D_sp, L_sp = build_physical_matrices_sparse(m_vector, 
                                                          m_gs, 
                                                          spring_Y, 
                                                          d_dashpot, 
                                                          d_drag)

        # 3. 对 normal 节点进行随机合并
        # 设置随机种子以保证可重复性
        rng = np.random.RandomState(self.seed)
        
        # 计算要保留的目标节点数量
        target_count = max(1, int(n_normal * self.ratio))
        
        # 从 normal 节点中随机选择目标节点索引
        target_indices = rng.choice(n_normal, size=target_count, replace=False)
        target_indices.sort()
        
        # 创建映射数组：每个 normal 节点映射到一个目标节点
        # -1 表示该节点是目标节点本身
        mapping = np.full(n_normal, -1, dtype=np.int64)
        
        # 将非目标节点随机分配到目标节点
        non_target_indices = np.setdiff1d(np.arange(n_normal), target_indices)
        if len(non_target_indices) > 0:
            # 随机分配到各个目标节点
            assignments = rng.randint(0, target_count, size=len(non_target_indices))
            for i, nt_idx in enumerate(non_target_indices):
                mapping[nt_idx] = target_indices[assignments[i]]
        
        # 4. 构建投影矩阵 P [n x n]
        P = np.eye(n)
        
        # 设置非目标节点的映射：将这些节点合并到对应的目标节点
        for i, normal_idx in enumerate(normal_indices):
            if mapping[i] >= 0:
                # 该节点需要合并到目标节点
                target_normal_idx = normal_indices[mapping[i]]
                P[normal_idx, target_normal_idx] = 1.0
                P[normal_idx, normal_idx] = 0.0

        # import pdb; pdb.set_trace()

        # # 5. 计算降维后的物理矩阵
        # M_hat = P.T @ M_sp @ P
        # D_hat = P.T @ D_sp @ P
        # L_hat = P.T @ L_sp @ P

        elapsed = time() - st_global
        actual_reduced = n - len(non_target_indices) if len(non_target_indices) > 0 else n
        logging.info(f"Random Reduction done: {n} nodes ({n_ctrl} ctrl) -> {actual_reduced} nodes. "
                     f"Normal nodes compressed from {n_normal} to {target_count}. "
                     f"Time: {elapsed:.3f}s")

        return P
