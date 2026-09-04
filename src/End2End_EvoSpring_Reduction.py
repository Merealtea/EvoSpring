from models import ModelGeneral
import torch
from End2End_reduction_model import End2EndReduction
from qqtt.utils import logger, cfg
from core_model import MLP
"""
This model works for mechanical information estimations of spring-mass systems.
"""

class End2EndReduction_EvoSpring(ModelGeneral):
    def __init__(self, pos_dim, ld, layer_num, pre_layer_num, bottom_layer_num, mlp_hidden_layer, MP_times, enhance, agg_conv_pos, default_spring_Y, default_drag_damping=None, default_dashpot_damping=None, default_collision_elas=None, default_collision_fric=None, default_collision_object_elas=None, default_collision_object_fric=None):
        # in: d_x(used for driven nodes only),type
        # out: d_x
        in_dim = 3 + 1 + 1 + 60# init_pos, node_mass, node_damping, node_type, pos_encoding
  
        self.lagrangian = False
        super(ModelGeneral, self).__init__()
        edge_set_num = 1

        self.encode = MLP(in_dim, ld, ld, mlp_hidden_layer, True)
        
        # layer_num = 1
        # pre_layer_num = 1
        self.process = End2EndReduction(layer_num, pre_layer_num, bottom_layer_num, ld, mlp_hidden_layer, pos_dim,
                                    self.lagrangian, enhance, agg_conv_pos, edge_set_num,
                                    transformer_hidden_dim=128, transformer_num_layers=2, pos_encoding_dim=30)
        
        # 用于第一层之后预测标量 bias 的 decoder（每层一个独立的 MLP）
        # 将 edge_feature (1*M*C) 压缩为 1*C，然后预测一个标量
        self.edge_decode = torch.nn.ModuleList([
            MLP(ld, ld, 1, 3, False)
            for _ in range(layer_num)
        ])

        self.MP_times = MP_times
        self.pos_dim = pos_dim
        self.gamma = 0.5

        self.temp = torch.tensor(0.1)
        self.S_0 = default_spring_Y

        # Store default damping values
        self.drag_damping_0 = float(default_drag_damping) if default_drag_damping is not None else 0.5
        self.dashpot_damping_0 = float(default_dashpot_damping) if default_dashpot_damping is not None else 0.5

        # --- 新增的四个参数 Decoders ---
        # Learnable damping bias parameters (initialized to 0)
        self.drag_damping_bias = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32)).to(cfg.device) 
            for _ in range(layer_num)
        ])

        self.dashpot_damping_bias = torch.nn.ParameterList([
            torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32)).to(cfg.device) 
            for _ in range(layer_num)
        ])

        self.spring_Y_basis = []
        self.drag_damping_basis = []
        self.dashpot_damping_basis = []

    def load_warp_simulator(self, simulator):
        self.simulator = simulator

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def _get_pos_type(self, node_in):
        # 0:3 current_pos, 3:6 next_pos 6:9 mesh_pos, 9:12 node_vel 12:13 node_mass 13:16 node_damping, 16: type
        pos_mat_world = node_in[..., :self.pos_dim].clone() 

        node_mass = node_in[..., -2:-1].clone()

        # torch.cat((node_in[..., self.pos_dim:2 * self.pos_dim], node_in[..., :self.pos_dim]), dim=-1)
        node_type = node_in[..., -1].clone()
        return pos_mat_world, node_mass, node_type

    def _get_vel(self, node_in):
        return node_in[..., self.pos_dim * 3 : self.pos_dim * 4]

    def _pre(self, node_in, node_tar, node_type):
        # 0 : object node, 1 : surface node, 2: interior node, 3: controller node
        controller_node = (node_type == 3).bool()
        object_node = (node_type == 0).bool()

        # 0:3 current_pos, 3:6 next_pos 6:9 mesh_pos, 9:12 node_vel 12:13 node_mass 13:16 node_damping, 16: type
        node_in[controller_node][..., self.pos_dim:2*self.pos_dim] = node_tar[controller_node]
        node_in[object_node][..., self.pos_dim:2*self.pos_dim] = node_tar[object_node]
        return node_in

    def _mask(self, node_in, node_tar, node_type, node_predict):
        # only measure int nodes(0)
        int_node = (node_type == 0).bool().unsqueeze(-1)
        # assert int_node[0].sum() == 1577
        mask = torch.where(int_node, torch.ones_like(node_tar), torch.zeros_like(node_tar))
        node_predict = torch.where(int_node, node_predict, node_tar)
        return node_predict, mask

    def _get_nodal_latent_input(self, node_in):
        # in_dim for nodal encoding: [world, mesh_pos, node_mass, velocity, node_damping, type] out of [world, mesh_pos, node_mass, velocity, node_damping, type]
        return node_in
    
    def _get_mesh_pos(self, node_info):
        return node_info[..., : self.pos_dim].clone()

    def _EMD(self, node_feature, m_ids, m_gs, m_proj, m_pos, m_node_mass, m_node_type, object_radius=None):
        node_feature = self._get_nodal_latent_input(node_feature)
     
        x = self.encode(node_feature)
       
        mlvl_edge_feature, pooling_losses, downsample_results = self.process(x, m_ids, m_gs, m_pos, m_node_mass, m_node_type, self.temp, m_proj, object_radius)

        for key in downsample_results:
            if key in ['down_ps','down_mass' ,'down_type']:
                for idx, info in enumerate(downsample_results[key]):
                    downsample_results[key][idx] = info[0]
        return mlvl_edge_feature, pooling_losses, downsample_results
    
    def calculate_proj_L_D(self, P, edge, prev_spring_Y, prev_dashpot_damping, prev_damping):

        # ====================================================================
        # 构建 L_hat, D_hat
        # ====================================================================
        # 提取上一层的投影矩阵 P 和 边索引
        N_prev = P.shape[0]
        u_prev = edge[0]
        v_prev = edge[1]

        prev_spring_Y = torch.cat([prev_spring_Y, prev_spring_Y]) # bidirection

        # 1. 构造上一层的 L_prev
        L_prev = torch.zeros((N_prev, N_prev), device=P.device, dtype=P.dtype)
        L_prev[u_prev, v_prev] = -prev_spring_Y
        
        # 巧妙利用 scatter_add_ 快速计算 Laplacian 对角线（非对角线的相反数之和）
        diag_L = torch.zeros(N_prev, device=P.device, dtype=P.dtype)
        diag_L.scatter_add_(0, u_prev, prev_spring_Y)
        L_prev[torch.arange(N_prev), torch.arange(N_prev)] = diag_L

        # 2. 构造上一层的 D_prev
        D_prev = torch.zeros((N_prev, N_prev), device=P.device, dtype=P.dtype)
        
        # --- 2.1 相对阻尼 (Dashpot) -> 拉普拉斯形式 ---
        if prev_dashpot_damping.numel() == 1:
            dashpot_vec = prev_dashpot_damping.expand(u_prev.size(0))
        else:
            dashpot_vec = prev_dashpot_damping
            
        D_prev[u_prev, v_prev] = -dashpot_vec
        
        diag_dashpot = torch.zeros(N_prev, device=P.device, dtype=P.dtype)
        diag_dashpot.scatter_add_(0, u_prev, dashpot_vec)

        # --- 2.2 绝对阻尼 (Drag) -> 纯对角线形式 ---
        if prev_damping.numel() == 1:
            drag_vec = prev_damping.expand(N_prev)
        else:
            drag_vec = prev_damping

        # Dashpot 的对角线加上 Drag 作为总阻尼对角线
        D_prev[torch.arange(N_prev), torch.arange(N_prev)] = diag_dashpot + drag_vec

        # ====================================================================
        # 3. [新增] 提取真正的 N x K 降维投影矩阵
        # ====================================================================
        # P 目前是 N x N，列和大于 0 的列即为被保留下来的代表节点
        # 使用 torch.nonzero 找到这些有效列的索引
        active_cols = torch.nonzero(P.sum(dim=0) > 0).squeeze(-1)
        
        # 提取出一个形状为 N x K 的投影矩阵
        P_reduced = P[:, active_cols]

        # ====================================================================
        # 4. 伽辽金投影得到本层的 L_hat 和 D_hat (此时为 K x K)
        # ====================================================================
        L_hat = P_reduced.T @ L_prev @ P_reduced
        D_hat = P_reduced.T @ D_prev @ P_reduced

        return L_hat, D_hat

    def forward(self, m_idx, m_gs, m_proj, m_node_pos, m_node_mass, m_node_type, node_in, object_radius=None, prev_best_mech_info=None):
        if self.temp > 0.1 :
            self.temp *= self.gamma
            self.temp = torch.clamp(self.temp, 0.1)

        mlvl_s_out = []
        mlvl_drag_damping_out = []
        mlvl_dashpot_damping_out = []

        # Process current frame through EMD
        mlvl_edge_feature, pooling_losses, downsample_results = \
            self._EMD(node_in, m_idx, m_gs, m_proj, m_node_pos, m_node_mass, m_node_type, object_radius)

        for lvl, edge_feature in enumerate(mlvl_edge_feature):
            # edge_feature shape: (1, M, C)
            edge_mech_in_bias = self.edge_decode[lvl](edge_feature)[0, :, 0]  # shape: (M,)
            
            if lvl > 0 and lvl <= len(prev_best_mech_info):
                prev_spring_Y = torch.exp(prev_best_mech_info[lvl-1]["log_spring_Y"])
                prev_damping = prev_best_mech_info[lvl-1]["drag_damping"]
                prev_dashpot_damping = prev_best_mech_info[lvl-1]["dashpot_damping"]

                # 构建L_hat, D_hat
                L_hat, D_hat = self.calculate_proj_L_D(downsample_results['down_proj'][lvl-1], downsample_results['down_gs'][lvl-1],
                                                       prev_spring_Y, prev_dashpot_damping, prev_damping)

                edges = downsample_results['down_gs'][lvl]
                num_edges = edges.shape[1]
                spring_Y_basis = self.S_0 # -L_hat[edges[0], edges[1]][:num_edges//2]
                dashpot_array = -D_hat[edges[0], edges[1]]
                new_dashpot_basis = self.dashpot_damping_0 # torch.mean(dashpot_array) 
                drag_array = torch.sum(D_hat, axis=1)
                new_drag_basis = self.drag_damping_0 # torch.mean(drag_array)

                # 将提取的参数转为 Tensor 保存
                drag_damping_out = new_drag_basis + self.drag_damping_bias[lvl] * 100
                dashpot_damping_out = new_dashpot_basis + self.dashpot_damping_bias[lvl] * 100
                s_out = ((spring_Y_basis + edge_mech_in_bias * 1e3))

            else:
            # 2. predict drag_damping bias
            # Return spring stiffness and damping parameters (default + learnable bias)
                drag_damping_out = self.drag_damping_0 + self.drag_damping_bias[lvl] * 100
                dashpot_damping_out = self.dashpot_damping_0 + self.dashpot_damping_bias[lvl] * 100
                s_out = ((self.S_0 + edge_mech_in_bias * 1e3))

            s_out = torch.clip(s_out, 1e-8, cfg.spring_Y_max)
            mlvl_s_out.append(s_out)

            drag_damping_out = torch.clip(drag_damping_out, 1, 20.0) 
            dashpot_damping_out = torch.clip(dashpot_damping_out, 10, 200.0) 
            mlvl_drag_damping_out.append(drag_damping_out)
            mlvl_dashpot_damping_out.append(dashpot_damping_out)
    
        return mlvl_s_out, mlvl_drag_damping_out, mlvl_dashpot_damping_out, downsample_results, pooling_losses
    