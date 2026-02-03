import torch
import datasets as cdata
import spring_mass_dataset
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import models as models
import spring_mass_model 
import math
import random
import numpy as np
import os
from qqtt.utils import logger, cfg
import warp as wp
import gc  # 记得在文件头部导入
import os
from time import time
# 3. 如果是 OffscreenRenderer，无头模式仍需保留
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

@wp.kernel
def accum_loss_kernel(loss_accum: wp.array(dtype=float), frame_loss: wp.array(dtype=float)):
    # 简单的累加操作: loss_accum[0] += frame_loss[0]
    wp.atomic_add(loss_accum, 0, frame_loss[0])

class SpringMassTrainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self._create_model()
        mdata = self._create_datset_offline(mode='train')
        self.mdata = mdata

        if "cloth" in mdata.object_case or "package" in mdata.object_case:
                cfg.load_from_yaml("configs/phystwin_configs/cloth.yaml")
        else:
            cfg.load_from_yaml("configs/phystwin_configs/real.yaml")

        cfg.device = next(self.model.parameters()).device
        
        # create warp simulator
        self.model.module.load_warp_simulator(
            dt = mdata.dt,
            init_vertices = mdata.mesh_pos,
            init_springs = mdata.cells,
            init_spring_Y = mdata.spring_Y[0],
            init_rest_lengths = mdata.spring_reset_length,
            init_masses = mdata.masses,
            num_object_springs = mdata.spring_Y.shape[0],
            init_masks = None,
            init_velocities = mdata.velocity,
            num_all_points = mdata.num_object_points,
            num_surface_points = mdata.num_surface_points,
            num_original_points = mdata.num_original_points,
            controller_points = mdata.controller_point,
            object_points = mdata.object_point,
            object_visibilities = mdata.object_visibilities,
            object_motions_valid = mdata.object_motions_valid,
            collide_elas = mdata.collide_elas,
            collide_fric = mdata.collide_fric,
            dashpot_damping = mdata.dashpot_damping,
            drag_damping = mdata.drag_damping,
            collide_object_elas = mdata.collide_object_elas,
            collide_object_fric = mdata.collide_object_fric,
            collision_dist = mdata.collision_dist,
            reverse_z = cfg.reverse_z,
            spring_Y_min = cfg.spring_Y_min,
            spring_Y_max = cfg.spring_Y_max,
            self_collision = cfg.self_collision,
            num_substeps = cfg.num_substeps,
            device=cfg.device,
        )
        
        # 初始化 Warp 优化器（仅在训练模式且 collision_learn 开启时）
        if cfg.collision_learn:
            # 确保 requires_grad=True
            self.torch_collide_elas = wp.to_torch(self.model.module.simulator.wp_collide_elas).requires_grad_(True)
            self.torch_collide_fric = wp.to_torch(self.model.module.simulator.wp_collide_fric).requires_grad_(True)
            self.torch_collide_object_elas = wp.to_torch(self.model.module.simulator.wp_collide_object_elas).requires_grad_(True)
            self.torch_collide_object_fric = wp.to_torch(self.model.module.simulator.wp_collide_object_fric).requires_grad_(True)
            
            self.warp_params_list = [self.torch_collide_elas, self.torch_collide_fric, 
                                     self.torch_collide_object_elas, self.torch_collide_object_fric]
        else:
            self.warp_params_list = []

        self.optimizer = torch.optim.Adam([p for p in self.model.module.parameters()] + self.warp_params_list, 
                                          lr=self.args.lr, 
                                          betas=(0.9, 0.99),
                                          weight_decay=self.args.weight_decay)


        def linear_warmup_lr(epoch):
            max_lr = self.args.lr  
            min_lr = max_lr / 10
            if epoch < self.args.warmup_epochs:
                return min_lr + (max_lr - min_lr) * (epoch / self.args.warmup_epochs) 
            else:
                return 1.0  
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_warmup_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.gamma)
        if dist.get_rank() == 0:
            self.writer = SummaryWriter(os.path.join(self.args.dump_dir, 'log'))
        self.pbar = tqdm(range(self.current_epoch, self.args.num_epochs), unit="iters")

        os.makedirs(self.args.dump_dir, exist_ok=True)
        for subdir in ['ckpts', 'log', 'test_RMSE', 'spring_mech_info']:
            dir = os.path.join(self.args.dump_dir, subdir)
            os.makedirs(dir, exist_ok=True)

    def _create_model(self):
        if self.args.case == 'spring_mass':
            self.model_class = spring_mass_model.SpringMass
            self.dataset_class = spring_mass_dataset.MeshSpringMassDataset
            self.object_case = self.args.object_case
        else:
            raise NotImplementedError("A Case not wrapped yet")
        self.model = self.model_class(
            pos_dim=self.args.space_dim, 
            ld=self.args.hidden_dim, 
            layer_num=self.args.multi_mesh_layer, 
            pre_layer_num=self.args.pre_layer_num, 
            bottom_layer_num=self.args.bottom_layer_num,
            mlp_hidden_layer=self.args.hidden_depth, 
            MP_times=self.args.mp_time,
            enhance=self.args.enhance, 
            agg_conv_pos=self.args.agg_conv_pos
        )

        if self.args.restart_epoch < 0 and not self.args.scratch:
            try:
                self.model.load_state_dict(
                    torch.load(self.args.path),
                    strict=False
                )
            except:
                checkpoint = torch.load(self.args.path)
                model_dict = self.model.state_dict()
                # Filter out unnecessary keys in checkpoint
                checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
                keys = set(checkpoint.keys())
                # Check if shapes match
                for k in keys:
                    if model_dict[k].shape != checkpoint[k].shape:
                        print(f"\033[31mIgnoring parameter {k} due to shape mismatch:\033[0m {model_dict[k].shape} != {checkpoint[k].shape}")
                        del checkpoint[k]
                self.model.load_state_dict(checkpoint, strict=False)
            print(f"\033[31mLoad pretrained from\033[0m {self.args.path}")

        POST_FIX_1 = '_layernum_' + str(self.args.multi_mesh_layer)
        POST_FIX_2 = POST_FIX_1 + '_MPHIDDENLAYER_' + str(self.args.hidden_depth) + '_MPHIDDENTDIM_' + str(self.args.hidden_dim) + '_MPtime_' + str(self.args.mp_time) + '_NoiseLevel_' + str(
            self.args.noise_level)
        self.checkpt_name = self.args.case + POST_FIX_2 + '.pt'
        if self.args.restart_epoch >= 0:
            print('hi, restarting')
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(torch.load(os.path.join(self.args.dump_dxir, 'ckpts', str(self.args.restart_epoch) + "_" + self.checkpt_name)))
            else:
                self.model.load_state_dict(torch.load(os.path.join(self.args.dump_dir, 'ckpts', str(self.args.restart_epoch) + "_" + self.checkpt_name)))
            
            self.current_epoch = self.args.restart_epoch + 1
            if self.current_epoch < self.args.warmup_epochs:
                min_lr = self.args.lr / 10
                max_lr = self.args.lr
                self.args.lr = min_lr + (max_lr - min_lr) * (self.current_epoch / self.args.warmup_epochs)
            else:
                decay_epochs = self.current_epoch - self.args.warmup_epochs
                self.args.lr *= self.args.gamma**(decay_epochs)
                self.args.lr = max(self.args.lr, 1e-6)
            print('restarted lr is: ', self.args.lr)
        else:
            self.current_epoch = 0

        self.model = nn.parallel.DistributedDataParallel(
            self.model.cuda(self.args.local_rank),
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            find_unused_parameters=True  # [关键修复] 允许部分参数不参与计算
        )

    def _preproc_multi_infos(self, mdata, node_in_feature, edge_mech_in_feature, node_tar):
        # process the multi-level mesh for batched data here
    
        # no contact, then share the graph between batches
        # only need to reshape input tensor
        m_ids = mdata.m_idx
        m_gs_list = mdata.m_g
        m_gs_parents_list = mdata.m_edge_parents
        m_gs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in m_gs_list]
        m_gs_parents = [torch.tensor(g_parent, dtype=torch.long).to(self.device) for g_parent in m_gs_parents_list]
        
        # Load data to GPU
        node_in_feature = node_in_feature.to(self.device)
        edge_mech_in_feature = edge_mech_in_feature.to(self.device)
        node_tar = node_tar.to(self.device)

        return m_ids, m_gs, m_gs_parents, node_in_feature, edge_mech_in_feature, node_tar

    def _create_datset_offline(self, mode='train', stride=1):
        if mode == 'train':
            prob = np.random.rand()
            add_noise = (prob < 0.667)
            mdata = self.dataset_class(self.args.data_dir,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=stride,
                                       noise_shuffle=add_noise,
                                       noise_level=self.args.noise_level,
                                       noise_gamma=self.args.noise_gamma,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh,
                                       object_case=self.args.object_case,
                                       train_frame=self.args.train_frame,
                                       test_frame=self.args.test_frame,
                                       args=self.args)
        else:
            mdata = self.dataset_class(self.args.data_dir,
                                       layer_num=self.args.multi_mesh_layer,
                                       stride=stride,
                                       noise_shuffle=False,
                                       recal_mesh=self.args.recal_mesh,
                                       consist_mesh=self.args.consist_mesh,
                                       mode=mode,
                                       object_case=self.args.object_case,
                                       train_frame=self.args.train_frame,
                                       test_frame=self.args.test_frame,
                                       args=self.args)
        return mdata
    
    def generate_data_point_sequence(self, update_frame_num, enable_backward=False, set_object_point=True):
        simulator = self.model.module.simulator
        vertices_sequence = []
        velocities_sequence = []

        # [修改] 定义 Warp 端的 Loss 累加器
        loss_accum = None
        tape = None

        torch_device = cfg.device
        if isinstance(torch_device, torch.device):
             if torch_device.type == 'cuda':
                 # 转换为 "cuda:0", "cuda:1" 等
                 wp_device_str = f"cuda:{torch_device.index if torch_device.index is not None else 0}"
             else:
                 wp_device_str = "cpu"
        else:
             wp_device_str = str(torch_device)
        
        # 将 cfg.device 更新为 Warp 兼容的字符串或直接获取 Warp device 对象
        # 建议直接获取 Warp device 对象以避免后续歧义
        warp_device = wp.get_device(wp_device_str)
        
        if enable_backward:
            loss_accum = wp.zeros(1, dtype=float, device=warp_device, requires_grad=True)
            # [修改] 开始录制 Tape
            tape = simulator.tape
        
        # 初始状态记录 ... (保持不变)
        simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)
        vertices_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[0].wp_x).clone(), wp.to_torch(simulator.wp_states[0].wp_control_x).clone()]))
        velocities_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[0].wp_v).clone(), wp.to_torch(simulator.wp_states[0].wp_control_v).clone()]))

        # [修改] 将循环包裹在 Tape 上下文中（如果开启反向传播）
        # 注意：Warp 的 Tape 需要包裹整个仿真过程
        context_manager = tape if enable_backward else open(os.devnull) # 简单的上下文占位符
        
        with context_manager: 
            for frame_idx in range(1, update_frame_num):
                simulator.set_controller_target(frame_idx, pure_inference=not set_object_point)
                
                # Record data
                vertices_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[-1].wp_x).clone(), wp.to_torch(simulator.wp_states[-1].wp_control_x).clone()]))
                velocities_sequence.append(torch.cat([wp.to_torch(simulator.wp_states[-1].wp_v).clone(), wp.to_torch(simulator.wp_states[-1].wp_control_v).clone()]))
                
                if simulator.object_collision_flag:
                    simulator.update_collision_graph()
                
                # 运行仿真步
                simulator.step()
                
                # 计算 Loss
                if enable_backward:
                    if cfg.data_type == "real":
                        simulator.calculate_loss()
                    else:
                        simulator.calculate_simple_loss()
                    
                    # [关键修改] 使用 Kernel 在 Warp 内部累加 Loss
                    # simulator.loss 通常是一个大小为1的数组
                    wp.launch(kernel=accum_loss_kernel, dim=1, inputs=[loss_accum, simulator.loss])

                simulator.set_init_state(simulator.wp_states[-1].wp_x, simulator.wp_states[-1].wp_v, pure_inference=True)
                if cfg.data_type == "real" and frame_idx > 1:
                    simulator.update_acc()
                    simulator.set_acc_count(True)

        # 转换输出张量 ... (保持不变)
        vertices_tensor = torch.stack(vertices_sequence, dim=0)
        velocities_tensor = torch.stack(velocities_sequence, dim=0)
        node_mass = torch.cat([wp.to_torch(simulator.wp_masses).clone(), torch.zeros(simulator.num_control_points, device=vertices_tensor.device)])
        spring_Y = wp.to_torch(simulator.wp_spring_Y).clone()
        
        spring_reset_length = wp.to_torch(simulator.wp_rest_lengths).clone()
        spring_dashpot_damping = simulator.dashpot_damping

        drag_damping = simulator.drag_damping

        if enable_backward:
            # [关键修改] 返回 tape 和 loss_accum (Warp 变量)，而不是 Python float
            return vertices_tensor, velocities_tensor, node_mass, spring_Y, spring_reset_length, spring_dashpot_damping, drag_damping, tape, loss_accum
        else:
            return vertices_tensor, velocities_tensor, node_mass, spring_Y, spring_reset_length, spring_dashpot_damping, drag_damping


    def run_epoch(self, epoch, 
                  mode='train', 
                  ):
        torch.cuda.memory._record_memory_history(max_entries=100000)

        if mode != 'train':
            self.model.eval()
        mean_loss_insts = 0
        count_insts = 0

        # single step simulation
        

        # optimization
        self.optimizer.zero_grad()

        # Update simulator with predicted mechanical properties
        # Note: You may need to adjust this based on actual mech_info structure
        st = time()
        # if spring_mech_info is not None and spring_mech_info.dim() == 2:  # Shape: (N, feature_dim)
        #     # Extract and assign predicted properties to simulator
        #     # This is a placeholder - adjust based on your model's output structure
        #     predicted_spring_Y = spring_mech_info[:, 0]  # Young's modulus
        #     self.model.module.simulator.wp_spring_Y = wp.from_torch(predicted_spring_Y.contiguous(), dtype=wp.float32)

        #     predicted_rest_length = spring_mech_info[:, 1]
        #     self.model.module.simulator.wp_rest_lengths = wp.from_torch(predicted_rest_length.contiguous(), dtype=wp.float32)
        
        # Raw Data preparation part, Using warp and init spring mech guess to generate data sequence for estimated surface and interior points
        train_node_pos, train_node_vel, train_node_mass, train_log_spring_Y,\
              spring_reset_length, spring_dashpot_damping, drag_damping =\
                  self.generate_data_point_sequence(self.mdata.train_frame) 

        # Print spring_Y average and simulator parameters
        print(f"\n{'='*60}")
        print(f"Spring_Y Average: {train_log_spring_Y.mean().item():.6f}")
        print(f"Spring Dashpot Damping: {spring_dashpot_damping}")
        print(f"Drag Damping: {drag_damping}")
        print(f"\nCollision Parameters:")
        print(f"  collide_elas: {wp.to_torch(self.model.module.simulator.wp_collide_elas).item():.6f}")
        print(f"  collide_fric: {wp.to_torch(self.model.module.simulator.wp_collide_fric).item():.6f}")
        print(f"  collide_object_elas: {wp.to_torch(self.model.module.simulator.wp_collide_object_elas).item():.6f}")
        print(f"  collide_object_fric: {wp.to_torch(self.model.module.simulator.wp_collide_object_fric).item():.6f}")
        print(f"{'='*60}\n")

        # Data Process
        node_in_feature, edge_mech_in_feature, node_tar = self.mdata._preprocess(
            train_node_pos, train_node_vel, train_node_mass,
              train_log_spring_Y, spring_reset_length,
              spring_dashpot_damping, drag_damping
        )

        m_ids, m_gs, m_gs_parent, node_in_feature, edge_mech_in_feature, node_tar =\
              self._preproc_multi_infos(self.mdata, node_in_feature, edge_mech_in_feature, node_tar)

        # Encoder part
        mech_info = self.model(m_ids, m_gs, m_gs_parent, node_in_feature, edge_mech_in_feature, node_tar)
        
        # 解析预测值
        predicted_log_spring_Y = mech_info[:, 0]
        predicted_rest_length = mech_info[:, 1]
        
        # [关键] 映射到 Warp 时，必须设置 requires_grad=True
        # 这样 Warp 才会为这些数组分配梯度缓冲区，供 Tape 使用
        wp_predicted_spring_Y = wp.from_torch(predicted_log_spring_Y.contiguous(), dtype=wp.float32, requires_grad=True)
        wp_predicted_rest_length = wp.from_torch(predicted_rest_length.contiguous(), dtype=wp.float32, requires_grad=True)
        
        self.model.module.simulator.wp_spring_Y = wp_predicted_spring_Y
        self.model.module.simulator.wp_rest_lengths = wp_predicted_rest_length

        print("Forward time is : {}".format(time() - st))
        st = time()
        # 运行仿真
        if mode == 'train':
            # 获取 tape 和 loss_accum
            pos, vel, _, _, _, _, _, tape, loss_accum = self.generate_data_point_sequence(
                update_frame_num=self.mdata.train_frame,
                enable_backward=True,
                set_object_point=False
            )

            # 获取 Loss 的数值用于统计 (float)
            loss_val = wp.to_torch(loss_accum).item()

            # -------------------------------------------------------
            # [核心修复] 手动梯度桥接流程
            # -------------------------------------------------------
            
            # 1. Warp 反向传播: 计算 loss_accum 对 simulator.wp_spring_Y 的梯度
            tape.backward(loss=loss_accum)

            # 2. 提取 Warp 计算出的梯度，转为 PyTorch 张量
            # 注意：如果 Warp 梯度为 None (未参与计算)，则设为 0
            grad_spring_Y = wp.to_torch(wp_predicted_spring_Y.grad).clone() if wp_predicted_spring_Y.grad else torch.zeros_like(predicted_spring_Y)
            grad_rest_length = wp.to_torch(wp_predicted_rest_length.grad).clone() if wp_predicted_rest_length.grad else torch.zeros_like(predicted_rest_length)
            
            # 3. 将梯度手动注入 PyTorch 计算图
            # 我们对模型输出的 predicted_spring_Y 调用 backward，并传入 Warp 算出的梯度
            torch.autograd.backward(
                tensors=[predicted_log_spring_Y, predicted_rest_length],
                grad_tensors=[grad_spring_Y, grad_rest_length]
            )

            # 4. PyTorch 优化器更新神经网络参数
            self.optimizer.step()
            print(f"Elas Grad: {self.torch_collide_elas.grad}")
            print(f"Elas Grad: {self.torch_collide_fric.grad}")
            print(f"Elas Grad: {self.torch_collide_object_elas.grad}")
            print(f"Elas Grad: {self.torch_collide_object_fric.grad}")
            import pdb; pdb.set_trace()
            
        else:
            # 验证模式不需要 tape
            _, _, _, _, _, _, _, loss = self.generate_data_point_sequence(
                update_frame_num=self.mdata.train_frame,
                enable_backward=False,
                set_object_point=False
            )
            loss_val = loss if isinstance(loss, float) else loss # 处理验证集返回值

        print("Backward time : {}".format(time() - st))
        # stats
        mean_loss = loss_val
        count = 1  # Number of samples in this batch

        # stats
        with torch.autograd.no_grad():
            mean_loss_insts += mean_loss
            count_insts += count

        # stats
        mean_loss_insts /= count_insts
        if mode == 'train':
            # opt scheduler
            if epoch < self.args.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                if self.optimizer.param_groups[0]['lr'] > 1e-6:
                    self.scheduler.step()
                    if self.optimizer.param_groups[0]['lr'] < 1e-6:
                        self.optimizer.param_groups[0]['lr'] = 1e-6
        else:
            self.model.eval()

        final_mech_info = mech_info.detach()
        return mean_loss_insts, final_mech_info

    
    def train(self):
        # load phystwin zero-grad optimization results and warp simulator here
        best_loss = 1e8
        best_mech_info = None
        best_epoch = None
        mech_info = None

        for epoch in self.pbar:
            mean_loss_train, mech_info = self.run_epoch(epoch, 
                                                        mech_info,
                                                        'train')

            with torch.autograd.no_grad():
                # train loss record
                if dist.get_rank() == 0:
                    self.pbar.set_description(f"Epoch: {epoch}, Training RMSE: {math.sqrt(mean_loss_train)}")
                    self.writer.add_scalar('RMSE/train', math.sqrt(mean_loss_train), epoch)
                    self.writer.add_scalar('lr/train', self.optimizer.param_groups[0]['lr'], epoch)
                    # dump ckpt
                    ckpt_path = os.path.join(self.args.dump_dir, 'ckpts', str(epoch) + "_" + self.checkpt_name)
                    torch.save(self.model.module.state_dict(), ckpt_path)
                    
                    # dump spring emch info
                    mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', str(epoch) + "_" + self.checkpt_name)
                    torch.save(mech_info.clone(), mech_info_path)

                    if best_loss > mean_loss_train:
                        best_mech_info = mech_info.clone()
                        best_epoch = epoch
                        best_loss = mean_loss_train

                # valid loss record
                if self.args.n_valid > 0 and (epoch+1) % 5 == 0:
                    mean_loss_valid = self.run_epoch(epoch, 
                                                    mech_info,
                                                    'test',
                                                    self.mdata)
                    if dist.get_rank() == 0:
                        self.writer.add_scalar('RMSE/test', math.sqrt(mean_loss_valid), epoch)
                dist.barrier()
        
        # dump best spring emch info estimation
        mech_info_path = os.path.join(self.args.dump_dir, 'spring_mech_info', "best_{}.pth".format(best_epoch))
        torch.save(best_mech_info, mech_info_path)
    
    def test(self, epoch=None):
        self.model.eval()
        epoch = self.args.restart_epoch if epoch==None else epoch
        rmse = []
        with torch.autograd.no_grad():
            # reload the model
            print('test on epoch, ', epoch)
            if self.args.n_test > 0:
                mean_loss_test = self.run_epoch(epoch, mode='test')
                print('epoch', epoch, 'RMSE/test', math.sqrt(mean_loss_test))
                if dist.get_rank() == 0:
                    self.writer.add_scalar('RMSE/test', math.sqrt(mean_loss_test), epoch)

            rmse.append(math.sqrt(mean_loss_test))

        if dist.get_rank() == 0:
            RMSE_cell = np.array([np.mean(rmse)])
            dump_path = os.path.join(self.args.dump_dir, 'test_RMSE', 'epoch_' + str(epoch) + '.csv')
            np.savetxt(dump_path, RMSE_cell, delimiter=',')
        
    
    def rollout(self, epoch=None, time_stps=None):
        epoch = self.args.restart_epoch if epoch == None else epoch
        instance_list = list(range(self.args.n_test))
        errors = []
        self.model.eval()
        with torch.autograd.no_grad():
            for i in tqdm(range(len(instance_list))):
                id = instance_list[i]
                # rollout for this instance, then record into a file
                mdata = self._create_datset_offline(id, mode='test')
                if time_stps is None:
                    L = mdata.in_feature.shape[0]
                    instance_rollout_error = np.zeros(L)
                    time_stps = L
                else:
                    L = time_stps
                    instance_rollout_error = np.zeros(L)
                for id_batch, b_data in enumerate(DataLoader(mdata, batch_size=1, shuffle=False)):
                    # print('id_batch', id_batch)
                    _, n, _ = mdata.in_feature.shape
                    m_ids = mdata.m_idx
                    m_gs_list = mdata.m_g
                    m_gs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in m_gs_list]
                    if mdata.has_contact:
                        m_cgs = [torch.tensor(g, dtype=torch.long).to(self.device) for g in mdata.m_cgs[id_batch]]
                        m_g_cg = [[g, cg] for g, cg in zip(m_gs, m_cgs)]
                        m_gs = m_g_cg
                    pen_coeff = mdata.suggested_pen_coef().to(self.device)
                    if id_batch == 0:
                        current_stat = mdata[0].x.reshape(1, n, -1).to(self.device)
                    
                    b_data.y = b_data.y.reshape(1, n, -1).to(self.device)
                    loss, out, _ = self.model(m_ids, m_gs, current_stat, b_data.y, pen_coeff)
                    # record global error
                    instance_rollout_error[id_batch] = math.sqrt(loss.item())
                    # push forward state
                    current_stat = mdata._push_forward(out, current_stat)
                    current_stat = current_stat.detach()

                dir = os.path.join(self.args.dump_dir, f'rollout_RMSE_epoch_' + str(epoch))
                os.makedirs(dir, exist_ok=True)
                dump_path = os.path.join(dir, str(id) + '.csv')
                np.savetxt(dump_path, instance_rollout_error, delimiter=',')
                errors.append(np.mean(instance_rollout_error))

    def visualize_simulation(self, mech_info_path, output_video_path='simulation_visualization.mp4'):
        """
        Run pure inference simulation and create PyVista visualization video.

        Args:
            mech_info_path: Path to the best_mech_info .pth file
            output_video_path: Path to save the output video
        """
        import pyvista as pv
        import h5py

        self.model.eval()

        with torch.autograd.no_grad():
            # Load dataset
            mdata = self._create_datset_offline(mode='train')

            # Load config
        if "cloth" in mdata.object_case or "package" in mdata.object_case:
            cfg.load_from_yaml("configs/phystwin_configs/cloth.yaml")
        else:
            cfg.load_from_yaml("configs/phystwin_configs/real.yaml")

        cfg.device = next(self.model.parameters()).device

        # Load warp simulator
        self.model.module.load_warp_simulator(
            dt=mdata.dt,
            init_vertices=mdata.mesh_pos,
            init_springs=mdata.cells,
            init_spring_Y=mdata.spring_Y[0],
            init_rest_lengths=mdata.spring_reset_length,
            init_masses=mdata.masses,
            num_object_springs=mdata.spring_Y.shape[0],
            init_masks=None,
            init_velocities=mdata.velocity,
            num_all_points=mdata.num_object_points,
            num_surface_points=mdata.num_surface_points,
            num_original_points=mdata.num_original_points,
            controller_points=mdata.controller_point,
            object_points=mdata.object_point,
            object_visibilities=mdata.object_visibilities,
            object_motions_valid=mdata.object_motions_valid,
            collide_elas=mdata.collide_elas,
            collide_fric=mdata.collide_fric,
            dashpot_damping=mdata.dashpot_damping,
            drag_damping=mdata.drag_damping,
            collide_object_elas=mdata.collide_object_elas,
            collide_object_fric=mdata.collide_object_fric,
            collision_dist=mdata.collision_dist,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            self_collision=cfg.self_collision,
            num_substeps=cfg.num_substeps,
            device=cfg.device,
        )

        # Load best_mech_info
        logger.info(f"Loading mech_info from: {mech_info_path}")
        mech_info = torch.load(mech_info_path, map_location=cfg.device)

        # Update simulator with predicted mechanical properties
        if mech_info is not None and mech_info.dim() == 2:
            predicted_spring_Y = mech_info[:, 0]
            self.model.module.simulator.wp_spring_Y = wp.from_torch(predicted_spring_Y.contiguous(), dtype=wp.float32)

            predicted_rest_length = mech_info[:, 1]
            self.model.module.simulator.wp_rest_lengths = wp.from_torch(predicted_rest_length.contiguous(), dtype=wp.float32)

        # Run pure inference simulation and collect all points data
        simulator = self.model.module.simulator

        # Storage for visualization data
        object_points_sequence = []  # Simulated object points
        controller_points_sequence = []  # Controller points
        gt_object_points_sequence = []  # Ground truth object points

        # Initialize simulation
        simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)

        # Collect initial state
        object_points_sequence.append(wp.to_torch(simulator.wp_states[0].wp_x).detach().cpu().numpy())
        controller_points_sequence.append(wp.to_torch(simulator.wp_states[0].wp_control_x).detach().cpu().numpy())
        gt_object_points_sequence.append(mdata.object_point[0])

        # Run simulation
        mdata.train_frame = 58
        logger.info(f"Running pure inference simulation for {mdata.train_frame} frames...")
        for frame_idx in tqdm(range(1, mdata.train_frame)):
            # Set controller target with pure_inference=True
            simulator.set_controller_target(frame_idx, pure_inference=True)

            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            # Run simulation step
            simulator.step()

            # Collect data
            object_points_sequence.append(wp.to_torch(simulator.wp_states[-1].wp_x).detach().cpu().numpy())
            controller_points_sequence.append(wp.to_torch(simulator.wp_states[-1].wp_control_x).detach().cpu().numpy())
            gt_object_points_sequence.append(mdata.object_point[frame_idx])

            # Reset for next step
            simulator.set_init_state(simulator.wp_states[-1].wp_x, simulator.wp_states[-1].wp_v)
            if cfg.data_type == "real" and frame_idx > 1:
                simulator.update_acc()
                simulator.set_acc_count(True)

        logger.info("Simulation completed. Creating PyVista visualization...")
        # Start a virtual framebuffer (requires xvfb to be installed)
        pv.start_xvfb()

        # Create PyVista visualization
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        plotter.set_background('white')
        # Set camera view
        plotter.camera_position = 'xy'
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 30

        plotter.open_movie(output_video_path, framerate=30)
        
        # plotter.show(auto_close=False)  # 必须先 show，建立窗口上下文

        # import pdb; pdb.set_trace()
        # Visualize each frame
        for frame_idx in tqdm(range(len(object_points_sequence)), desc="Rendering frames"):
            plotter.clear()

            # Add simulated object points (green)
            obj_points = object_points_sequence[frame_idx]
            if len(obj_points) > 0:
                obj_cloud = pv.PolyData(obj_points)
                plotter.add_mesh(obj_cloud, color='green', point_size=8.0,
                                render_points_as_spheres=True, label='Simulated Object')

            # Add controller points (blue)
            ctrl_points = controller_points_sequence[frame_idx]
            if len(ctrl_points) > 0:
                ctrl_cloud = pv.PolyData(ctrl_points)
                plotter.add_mesh(ctrl_cloud, color='blue', point_size=10.0,
                                render_points_as_spheres=True, label='Controller')

            # Add ground truth object points (red, semi-transparent)
            gt_points = gt_object_points_sequence[frame_idx]
            if len(gt_points) > 0:
                gt_cloud = pv.PolyData(gt_points)
                plotter.add_mesh(gt_cloud, color='red', point_size=6.0,
                                opacity=0.5, render_points_as_spheres=True,
                                label='Ground Truth')

            # Add legend and text
            plotter.add_text(f'Frame: {frame_idx}/{len(object_points_sequence)-1}',
                            position='upper_left', font_size=12, color='black')

            # Write frame
            plotter.write_frame()

        plotter.close()
        logger.info(f"Video saved to: {output_video_path}")

        logger.info("Visualization complete!")

        return object_points_sequence, controller_points_sequence, gt_object_points_sequence