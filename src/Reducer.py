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
        # node_type == 3 为 controller
        ctrl_indices = np.where(node_type == 3)[0]
        normal_indices = np.where(node_type != 3)[0]
        
        n_ctrl = len(ctrl_indices)
        n_normal = len(normal_indices)
        
        if n_normal <= 1:
            logging.warning("No normal nodes to reduce or system too small.")
            return np.eye(n), None, None, None

        # 2. 构建物理矩阵 (使用你提供的 sparse 构建函数)
        M_sp, D_sp, L_sp = build_physical_matrices_sparse(m_vector, 
                                                          m_gs, 
                                                          spring_Y, 
                                                          d_dashpot, 
                                                          d_drag)

        # =================================================================
        # 3. 模态分析与投影空间计算 (全局系统分析)
        # =================================================================
        k_modes = min(self.num_modes, n - 2) 
        eigenvalues, Phi = eigsh(L_sp, k=k_modes, M=M_sp, sigma=-1e-4, which='LM')
        
        # M-正交归一化
        for i in range(k_modes):
            norm = np.sqrt(Phi[:, i].T @ M_sp @ Phi[:, i])
            if norm > 1e-12: Phi[:, i] /= norm

        # =================================================================
        # 4. Lyapunov 求解获取特征嵌入 Z
        # =================================================================
        # 识别受力节点 (与控制器相连的节点)
        u, v = m_gs[0], m_gs[1]
        connected_v = v[np.isin(u, ctrl_indices)]
        connected_u = u[np.isin(v, ctrl_indices)]
        target_nodes = np.unique(np.concatenate([connected_u, connected_v]))
        target_nodes = target_nodes[node_type[target_nodes] != 3]
        
        m_inputs = len(target_nodes) if len(target_nodes) > 0 else 1
        if len(target_nodes) == 0: target_nodes = np.array([0])

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
        
        P_r_tilde = solve_continuous_lyapunov(A_r - 1e-5 * np.eye(2 * k_modes), -B_r_eff @ B_r_eff.T)
        
        nu_r_T = np.block([[ones_n.T @ D_sp @ Phi, ones_n.T @ M_sp @ Phi]])
        beta_a = - (nu_r_T @ P_r_tilde @ nu_r_T.T)[0, 0] / (sigma_D ** 2)
        Pi_r = np.block([
            [Phi.T @ ones_n @ ones_n.T @ Phi, np.zeros((k_modes, k_modes))],
            [np.zeros((k_modes, k_modes)), np.zeros((k_modes, k_modes))]
        ])
        P_r_gramian = P_r_tilde + beta_a * Pi_r

        # 欧氏空间嵌入 Z (N x k)
        P_r11 = P_r_gramian[:k_modes, :k_modes]
        w, v_e = np.linalg.eigh(P_r11)
        w[w < 0] = 0
        C = v_e @ np.diag(np.sqrt(w))
        Z = Phi @ C

        # =================================================================
        # 5. 限制性聚类逻辑：仅聚类 normal_indices，且只允许空间相连的节点合并
        #    【新增约束】: 只合并 type 相同的点，并且不合并 controller points
        # =================================================================
        # 提取普通节点的坐标嵌入
        Z_normal = Z[normal_indices, :]
        
        # 获取普通节点的 type
        normal_types = node_type[normal_indices]
        
        # 构建普通节点之间的邻接关系（只考虑 normal 节点之间的连接）- 使用矩阵运算
        n_nodes = len(normal_indices)
        
        # 创建从全局索引到局部索引的映射数组
        global_to_local = np.full(n, -1, dtype=np.int32)
        global_to_local[normal_indices] = np.arange(n_nodes)
        
        # 向量化过滤：只保留两端都是 normal nodes 的边
        u, v = m_gs[0], m_gs[1]
        u_local = global_to_local[u]
        v_local = global_to_local[v]
        
        # 过滤掉自环和无效边（-1 表示不是 normal node）
        valid_mask = (u_local >= 0) & (v_local >= 0) & (u_local != v_local)
        u_local = u_local[valid_mask]
        v_local = v_local[valid_mask]
        
        # 使用向量化聚类方法
        cluster_labels = self._constrained_hierarchical_clustering_vectorized(
            Z_normal, u_local, v_local, n_nodes, normal_types
        )
        n_clusters_normal = len(np.unique(cluster_labels))

        # =================================================================
        # 6. 构建混合投影矩阵 P (N x (n_clusters_normal + n_ctrl))
        # =================================================================
        # P 的形状为 [原始节点数, 压缩后节点数]
        total_reduced_nodes = n_clusters_normal + n_ctrl
        P = np.zeros((n, total_reduced_nodes))

        # 第一部分：普通节点的聚类映射 (填充前 n_clusters_normal 列)
        for i, normal_idx in enumerate(normal_indices):
            cluster_id = cluster_labels[i]  # 标签从1开始转为0开始
            P[normal_idx, cluster_id] = 1.0

        # 第二部分：Controller 节点的一对一保留 (填充后续列)
        for i, ctrl_idx in enumerate(ctrl_indices):
            target_col = n_clusters_normal + i
            P[ctrl_idx, target_col] = 1.0

        # 7. 计算降维后的物理矩阵
        M_hat = P.T @ M_sp @ P
        D_hat = P.T @ D_sp @ P
        L_hat = P.T @ L_sp @ P

        print(f"Reduction done: {n} nodes ({n_ctrl} ctrl) -> {total_reduced_nodes} nodes. "
                     f"Normal nodes compressed from {n_normal} to {n_clusters_normal}.")

        return P, M_hat, D_hat, L_hat

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
        # =================================================================
        # 1. 向量化计算所有边的距离
        # =================================================================
        Z_src = Z_normal[edge_src]  # [n_edges, k]
        Z_dst = Z_normal[edge_dst]  # [n_edges, k]
        edge_distances = np.linalg.norm(Z_src - Z_dst, axis=1)  # [n_edges]
        
        # =================================================================
        # 2. 应用 type 约束：只保留 type 相同的边
        # =================================================================
        if node_types is not None:
            type_src = node_types[edge_src]
            type_dst = node_types[edge_dst]
            type_mask = (type_src == type_dst)
            
            # 过滤掉 type 不同的边
            edge_src = edge_src[type_mask]
            edge_dst = edge_dst[type_mask]
            edge_distances = edge_distances[type_mask]
        
        # =================================================================
        # 3. 计算自适应阈值
        # =================================================================
        if len(edge_distances) > 0:
            t_cutoff = np.percentile(edge_distances, 70)
        else:
            t_cutoff = float('inf')
        
        # =================================================================
        # 4. 构建约束邻接矩阵（只保留距离小于阈值的边）
        # =================================================================
        valid_mask = edge_distances <= t_cutoff
        valid_src = edge_src[valid_mask]
        valid_dst = edge_dst[valid_mask]
        
        # 构建稀疏邻接矩阵（对称）
        n_valid = len(valid_src)
        data = np.ones(n_valid * 2, dtype=np.float32)
        rows = np.concatenate([valid_src, valid_dst])
        cols = np.concatenate([valid_dst, valid_src])
        constrained_adj = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        
        # =================================================================
        # 5. 使用并查集/连通分量算法进行聚类
        #    这等价于贪心合并，但用高效的图算法实现
        # =================================================================
        n_components, labels = connected_components(constrained_adj, directed=False, return_labels=True)
        
        logging.info(f"Constrained clustering: {n_nodes} nodes -> {n_components} clusters, "
                     f"threshold={t_cutoff:.4f}, valid_edges={n_valid}")
        
        return labels
