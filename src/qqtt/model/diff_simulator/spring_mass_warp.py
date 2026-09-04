import torch
import numpy as np
from qqtt.utils import logger, cfg
import warp as wp

# Warp 1.14+ 使用 Logger 协议代替了旧的 warn 函数
# 通过自定义 Logger 屏蔽 "Running the tape backwards" 警告
class _SuppressTapeWarningsLogger:
    """包装默认 Logger，静默丢弃 tape backwards 警告。"""

    def __init__(self):
        self._default = wp._src.utils.LoggerBasic()

    def debug(self, message: str) -> None:
        self._default.debug(message)

    def info(self, message: str) -> None:
        self._default.info(message)

    def warning(self, message: str, category=None, stacklevel: int = 1) -> None:
        if "Running the tape backwards" in str(message):
            return
        self._default.warning(message, category, stacklevel + 1)

    def error(self, message: str) -> None:
        self._default.error(message)

wp.set_logger(_SuppressTapeWarningsLogger())

# wp.init()
# wp.set_device("cuda:0")
# if not cfg.use_graph:
#     wp.config.mode = "debug"
#     wp.config.verbose = True
#     wp.config.verify_autograd_array_access = True


class State:
    def __init__(self, wp_init_vertices, num_control_points):
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_collision = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_ground = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=True)
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)

        # --- 新增：用于在 Kernel 中记录碰撞信息的 Buffer ---
        # num_object_points = wp_init_vertices.shape[0]
        # self.wp_hit_count = wp.zeros(num_object_points, dtype=wp.int32, requires_grad=False)
        # self.wp_hit_indices = wp.zeros((num_object_points, 100), dtype=wp.int32, requires_grad=False)
        # self.wp_hit_impulses = wp.zeros((num_object_points, 100), dtype=wp.vec3, requires_grad=False)

    def clear_forces(self):
        self.wp_vertice_forces.zero_()
        # self.wp_hit_count.zero_()
        # # --- 加上下面这两行，每次都把显存洗干净 ---
        # self.wp_hit_indices.zero_()
        # self.wp_hit_impulses.zero_()

    @property
    def requires_grad(self):
        """Indicates whether the state arrays have gradient computation enabled."""
        return self.wp_x.requires_grad


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_int(data: wp.array(dtype=wp.int32), origin: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_float(data: wp.array(dtype=wp.float32), origin: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def set_control_points(
    num_substeps: int,
    original_control_point: wp.array(dtype=wp.vec3),
    target_control_point: wp.array(dtype=wp.vec3),
    step: int,
    control_x: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    t = float(step + 1) / float(num_substeps)
    control_x[tid] = (
        original_control_point[tid]
        + (target_control_point[tid] - original_control_point[tid]) * t
    )


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),
    num_object_points: int,
    springs: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    spring_Y: wp.array(dtype=float),
    dashpot_damping: wp.array(dtype=float),
    spring_Y_min: float,
    spring_Y_max: float,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if wp.exp(spring_Y[tid]) > spring_Y_min:

        idx1 = springs[tid][0]
        idx2 = springs[tid][1]

        if idx1 >= num_object_points:
            x1 = control_x[idx1 - num_object_points]
            v1 = control_v[idx1 - num_object_points]
        else:
            x1 = x[idx1]
            v1 = v[idx1]
        if idx2 >= num_object_points:
            x2 = control_x[idx2 - num_object_points]
            v2 = control_v[idx2 - num_object_points]
        else:
            x2 = x[idx2]
            v2 = v[idx2]

        rest = rest_lengths[tid]

        dis = x2 - x1
        dis_len = wp.length(dis)

        d = dis / wp.max(dis_len, 1e-6)

        spring_force = (
            wp.clamp(wp.exp(spring_Y[tid]), low=spring_Y_min, high=spring_Y_max)
            * (dis_len / rest - 1.0)
            * d
        )

        v_rel = wp.dot(v2 - v1, d)
        dashpot_forces = dashpot_damping[0] * v_rel * d

        overall_force = spring_force + dashpot_forces

        if idx1 < num_object_points:
            wp.atomic_add(f, idx1, overall_force)
        if idx2 < num_object_points:
            wp.atomic_sub(f, idx2, overall_force)


@wp.kernel
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: wp.array(dtype=float),
    reverse_factor: float,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v0 = v[tid]
    f0 = f[tid]
    m0 = masses[tid]

    drag_damping_factor = wp.exp(-dt * drag_damping[0])
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    v2 = v1 * drag_damping_factor

    v_new[tid] = v2


@wp.func
def loop(
    i: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    clamp_collide_object_elas: float,
    clamp_collide_object_fric: float,
    # # --- 新增参数 ---
    # hit_count: wp.array(dtype=wp.int32),
    # hit_indices: wp.array2d(dtype=wp.int32),
    # hit_impulses: wp.array2d(dtype=wp.vec3),
    sort_buffer: wp.array2d(dtype=wp.vec3), # 传入草稿纸
):
    x1 = x[i]
    v1 = v[i]
    m1 = masses[i]
    mask1 = masks[i]

    valid_count = float(0.0)
    idx = int(0)
    current_hits = int(0)

    for k in range(collision_number[i]):
        index = collision_indices[i][k]
        x2 = x[index]
        v2 = v[index]
        m2 = masses[index]
        mask2 = masks[index]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        # If the distance is less than the collision distance and the two points are moving towards each other
        # 碰撞条件保持不变
        if (
            mask1 != mask2
            and dis_len < collision_dist
            and wp.dot(dis, relative_v) < -1e-4
        ):
            valid_count += 1.0
            mass_inv_sum = 1.0 / m1 + 1.0 / m2

            # ==========================================
            # 修复 1：几何奇点 (完全重叠导致法线随机化)
            # ==========================================
            if dis_len < 1e-6:
                # 如果极度穿模，强制赋予一个 Z 轴向上的排斥法线，防止 NaN 或随机弹飞
                collision_normal = wp.vec3(0.0, 0.0, 1.0)
            else:
                collision_normal = dis / dis_len

            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            v_rel_n_length = wp.length(v_rel_n)
            
            # 法向冲量 (Normal Impulse)
            impulse_n = (-(1.0 + clamp_collide_object_elas) * v_rel_n) / mass_inv_sum

            # ==========================================
            # 修复 2：摩擦力奇点 (切向速度极小导致除零溢出)
            # ==========================================
            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.length(v_rel_t)

            if v_rel_t_length < 1e-6:
                # 几乎没有切向滑动，处于静摩擦状态，直接施加反向冲量抵消全部切向速度
                impulse_t = -v_rel_t / mass_inv_sum
            else:
                
                # 动摩擦状态：安全计算摩擦力圆锥截断 (Friction Cone Clamping)
                friction_limit = clamp_collide_object_fric * (1.0 + clamp_collide_object_elas) * v_rel_n_length
                # ratio 代表我们需要抵消多少比例的切向速度，最大为 1.0 (完全停止滑动)
                ratio = wp.min(friction_limit / v_rel_t_length, 1.0)
                impulse_t = -ratio * v_rel_t / mass_inv_sum

            J = impulse_n + impulse_t

            # # --- 新增：在这里直接将真实的碰撞 ID 和冲量写入显存 ---
            # # current_hits = hit_count[i]
            # if idx < 100:  
            #     hit_indices[i, idx] = index
            #     hit_impulses[i, idx] = J
            #     idx += 1
   
            sort_buffer[i, current_hits] = J
            current_hits += 1

    # 2. 对草稿纸中的冲量进行插入排序 (按模长平方从小到大)
    for step in range(1, current_hits):
        key_J = sort_buffer[i, step]
        key_mag = wp.length_sq(key_J)
        
        j = step - 1
        while j >= 0 and wp.length_sq(sort_buffer[i, j]) > key_mag:
            sort_buffer[i, j + 1] = sort_buffer[i, j]
            j -= 1
        
        sort_buffer[i, j + 1] = key_J

    # 3. 排序完成后，从最小的力开始累加，极大降低浮点数精度截断造成的误差
    J_sum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(current_hits):
        J_sum += sort_buffer[i, k]

    # hit_count[i] = int(idx)
    return valid_count, J_sum


@wp.kernel(enable_backward=False)
def update_potential_collision(
    x: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    grid: wp.uint64,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]
    mask1 = masks[i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index != i:
            x2 = x[index]
            mask2 = masks[index]

            dis = x2 - x1
            dis_len = wp.length(dis)
            # If the distance is less than the collision distance and the two points are moving towards each other
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = index
                collision_number[i] += 1


@wp.kernel
def object_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collide_object_elas: wp.array(dtype=float),
    collide_object_fric: wp.array(dtype=float),
    collision_dist: float,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    # # --- 新增参数 ---
    # hit_count: wp.array(dtype=wp.int32),
    # hit_indices: wp.array2d(dtype=wp.int32),
    # hit_impulses: wp.array2d(dtype=wp.vec3),
    sort_buffer: wp.array2d(dtype=wp.vec3), # <--- 接收草稿纸
    # # --------------
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    v1 = v[tid]
    m1 = masses[tid]

    clamp_collide_object_elas = wp.clamp(collide_object_elas[0], low=0.0, high=1.0)
    clamp_collide_object_fric = wp.clamp(collide_object_fric[0], low=0.0, high=2.0)

    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        collision_dist,
        clamp_collide_object_elas,
        clamp_collide_object_fric,
        # hit_count,     # 传入
        # hit_indices,   # 传入
        # hit_impulses,  # 传入
        sort_buffer, # <--- 传给 loop
    )

    if valid_count > 0:
        J_average = J_sum / valid_count
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1


@wp.kernel
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor

    x_z = x0[2]
    v_z = v0[2]
    next_x_z = (x_z + v_z * dt) * reverse_factor

    if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
        # Ground Collision
        v_normal = wp.dot(v0, normal) * normal
        v_tao = v0 - v_normal
        v_normal_length = wp.length(v_normal)
        v_tao_length = wp.max(wp.length(v_tao), 1e-6)
        clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
        clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

        v_normal_new = -clamp_collide_elas * v_normal
        a = wp.max(
            0.0,
            1.0
            - clamp_collide_fric
            * (1.0 + clamp_collide_elas)
            * v_normal_length
            / v_tao_length,
        )
        v_tao_new = a * v_tao

        v1 = v_normal_new + v_tao_new
        toi = -x_z / v_z
    else:
        v1 = v0
        toi = 0.0

    x_new[tid] = x0 + v0 * toi + v1 * (dt - toi)
    v_new[tid] = v1


@wp.kernel(enable_backward=False)
def compute_distances(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    distances: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if gt_mask[i] == 1:
        dist = wp.length(gt[i] - pred[j])
        distances[i, j] = dist
    else:
        distances[i, j] = 1e6


@wp.kernel(enable_backward=False)
def compute_neigh_indices(
    distances: wp.array2d(dtype=float),
    neigh_indices: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    min_dist = float(1e6)
    min_index = int(-1)
    for j in range(distances.shape[1]):
        if distances[i, j] < min_dist:
            min_dist = distances[i, j]
            min_index = j
    neigh_indices[i] = min_index


@wp.kernel
def compute_chamfer_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    neigh_indices: wp.array(dtype=wp.int32),
    loss_weight: float,
    chamfer_loss: wp.array(dtype=float),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        min_pred = pred[neigh_indices[i]]
        min_dist = wp.length(min_pred - gt[i])
        final_min_dist = loss_weight * min_dist * min_dist / float(num_valid)
        wp.atomic_add(chamfer_loss, 0, final_min_dist)


@wp.kernel
def compute_track_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    loss_weight: float,
    track_loss: wp.array(dtype=float),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        pred_x = pred[i][0]
        pred_y = pred[i][1]
        pred_z = pred[i][2]
        gt_x = gt[i][0]
        gt_y = gt[i][1]
        gt_z = gt[i][2]

        dist_x = wp.abs(pred_x - gt_x)
        dist_y = wp.abs(pred_y - gt_y)
        dist_z = wp.abs(pred_z - gt_z)

        if dist_x < 1.0:
            temp_track_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_track_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_track_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_track_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_track_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_track_loss_z = dist_z - 0.5

        temp_track_loss = temp_track_loss_x + temp_track_loss_y + temp_track_loss_z

        average_factor = float(num_valid) * 3.0

        final_track_loss = loss_weight * temp_track_loss / average_factor

        wp.atomic_add(track_loss, 0, final_track_loss)


@wp.kernel(enable_backward=False)
def set_int(input: int, output: wp.array(dtype=wp.int32)):
    output[0] = input


@wp.kernel(enable_backward=False)
def update_acc(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    prev_acc[tid] = v2[tid] - v1[tid]


@wp.kernel
def compute_acc_loss(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
    num_object_points: int,
    acc_count: wp.array(dtype=wp.int32),
    acc_weight: float,
    acc_loss: wp.array(dtype=wp.float32),
):
    if acc_count[0] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        tid = wp.tid()
        cur_acc = v2[tid] - v1[tid]
        cur_x = cur_acc[0]
        cur_y = cur_acc[1]
        cur_z = cur_acc[2]

        prev_x = prev_acc[tid][0]
        prev_y = prev_acc[tid][1]
        prev_z = prev_acc[tid][2]

        dist_x = wp.abs(cur_x - prev_x)
        dist_y = wp.abs(cur_y - prev_y)
        dist_z = wp.abs(cur_z - prev_z)

        if dist_x < 1.0:
            temp_acc_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_acc_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_acc_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_acc_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_acc_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_acc_loss_z = dist_z - 0.5

        temp_acc_loss = temp_acc_loss_x + temp_acc_loss_y + temp_acc_loss_z

        average_factor = float(num_object_points) * 3.0

        final_acc_loss = acc_weight * temp_acc_loss / average_factor

        wp.atomic_add(acc_loss, 0, final_acc_loss)


@wp.kernel
def compute_final_loss(
    chamfer_loss: wp.array(dtype=wp.float32),
    track_loss: wp.array(dtype=wp.float32),
    acc_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    loss[0] = chamfer_loss[0] + track_loss[0] + acc_loss[0]


@wp.kernel
def compute_simple_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    num_object_points: int,
    loss: wp.array(dtype=wp.float32),
):
    # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
    tid = wp.tid()
    pred_x = pred[tid][0]
    pred_y = pred[tid][1]
    pred_z = pred[tid][2]

    gt_x = gt[tid][0]
    gt_y = gt[tid][1]
    gt_z = gt[tid][2]

    dist_x = wp.abs(pred_x - gt_x)
    dist_y = wp.abs(pred_y - gt_y)
    dist_z = wp.abs(pred_z - gt_z)

    if dist_x < 1.0:
        temp_simple_loss_x = 0.5 * (dist_x**2.0)
    else:
        temp_simple_loss_x = dist_x - 0.5

    if dist_y < 1.0:
        temp_simple_loss_y = 0.5 * (dist_y**2.0)
    else:
        temp_simple_loss_y = dist_y - 0.5

    if dist_z < 1.0:
        temp_simple_loss_z = 0.5 * (dist_z**2.0)
    else:
        temp_simple_loss_z = dist_z - 0.5

    temp_simple_loss = temp_simple_loss_x + temp_simple_loss_y + temp_simple_loss_z

    average_factor = float(num_object_points) * 3.0

    final_simple_loss = temp_simple_loss / average_factor

    wp.atomic_add(loss, 0, final_simple_loss)


class SpringMassSystemWarp:
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        collision_dist=0.02,
        init_velocities=None,
        num_object_points=None,
        num_surface_points=None,
        num_original_points=None,
        controller_points=None,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        gt_object_points=None,
        gt_object_visibilities=None,
        gt_object_motions_valid=None,
        # 新增：未下采样的真值，用于 chamfer loss 计算
        gt_object_points_full=None,
        gt_object_visibilities_full=None,
        self_collision=False,
        disable_backward=False,
    ):
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        self.device = cfg.device

        # Record the parameters
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:num_object_points].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
        else:
            self.wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]

        self.dt = dt
        self.num_substeps = num_substeps
        self.wp_dashpot_damping = wp.from_torch(
            torch.tensor([dashpot_damping], dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.wp_drag_damping = wp.from_torch(
            torch.tensor([drag_damping], dtype=torch.float32, device=self.device),
            requires_grad=True,
        )

        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        if controller_points is None:
            assert num_object_points == self.n_vertices
        else:
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices

        self.num_object_points = num_object_points
        self.num_control_points = (
            controller_points.shape[1] if not controller_points is None else 0
        )
        self.controller_points = controller_points

        # Deal with the any collision detection
        self.object_collision_flag = 0
        if init_masks is not None:
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1

        if self_collision:
            assert init_masks is None
            self.object_collision_flag = 1
            # Make all points as the collision points
            init_masks = torch.arange(
                self.n_vertices, dtype=torch.int32, device=self.device
            )

        if self.object_collision_flag:
            self.wp_masks = wp.from_torch(
                init_masks[:num_object_points].int(),
                dtype=wp.int32,
                requires_grad=False,
            )

            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

        # Initialize the GT for calculating losses
        # 下采样后的真值用于 track loss
        self.gt_object_points = gt_object_points
        if cfg.data_type == "real":
            self.gt_object_visibilities = gt_object_visibilities.int()
            self.gt_object_motions_valid = gt_object_motions_valid.int()
        
        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points

        # 未下采样的真值用于 chamfer loss（如果提供了的话）
        self.gt_object_points_full = gt_object_points_full
        self.use_full_gt_for_chamfer = (gt_object_points_full is not None)
        if cfg.data_type == "real" and self.use_full_gt_for_chamfer:
            self.gt_object_visibilities_full = gt_object_visibilities_full.int()
            self.num_original_points_full = gt_object_points_full.shape[1]
        else:
            self.gt_object_visibilities_full = None
            self.num_original_points_full = self.num_original_points

        
        if num_original_points is None:
            self.num_original_points = self.num_object_points

        # # Do some initialization to initialize the warp cuda graph
        self.wp_springs = wp.from_torch(
            init_springs, dtype=wp.vec2i, requires_grad=False
        )
        self.wp_rest_lengths = wp.from_torch(
            init_rest_lengths, dtype=wp.float32, requires_grad=False
        )
        self.wp_masses = wp.from_torch(
            init_masses[:num_object_points], dtype=wp.float32, requires_grad=False
        )
        if cfg.data_type == "real":
            self.prev_acc = wp.zeros_like(self.wp_init_vertices, requires_grad=False)
            self.acc_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)

        self.wp_current_object_points = wp.from_torch(
            self.gt_object_points[1].clone(), dtype=wp.vec3, requires_grad=False
        )
        if cfg.data_type == "real":
            self.wp_current_object_visibilities = wp.from_torch(
                self.gt_object_visibilities[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_current_object_motions_valid = wp.from_torch(
                self.gt_object_motions_valid[0].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.num_valid_visibilities = int(self.gt_object_visibilities[1].sum())
            self.num_valid_motions = int(self.gt_object_motions_valid[0].sum())
            
            # 如果提供了 full GT，也初始化对应的 buffer
            if self.use_full_gt_for_chamfer:
                self.wp_current_object_points_full = wp.from_torch(
                    self.gt_object_points_full[1].clone(), dtype=wp.vec3, requires_grad=False
                )
                self.wp_current_object_visibilities_full = wp.from_torch(
                    self.gt_object_visibilities_full[1].clone(),
                    dtype=wp.int32,
                    requires_grad=False,
                )
                self.num_valid_visibilities_full = int(self.gt_object_visibilities_full[1].sum())

            self.wp_original_control_point = wp.from_torch(
                self.controller_points[0].clone(), dtype=wp.vec3, requires_grad=False
            )
            self.wp_target_control_point = wp.from_torch(
                self.controller_points[1].clone(), dtype=wp.vec3, requires_grad=False
            )

            self.chamfer_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.track_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.acc_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # Initialize the warp parameters
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = State(self.wp_init_velocities, self.num_control_points)
            self.wp_states.append(state)
        if cfg.data_type == "real":
            # 用于 track loss 的 distance matrix（下采样后的 GT）
            self.distance_matrix = wp.zeros(
                (self.num_original_points, self.num_surface_points), requires_grad=False
            )
            self.neigh_indices = wp.zeros(
                (self.num_original_points), dtype=wp.int32, requires_grad=False
            )
            # 如果提供了 full GT，也初始化对应的 buffer 用于 chamfer loss
            if self.use_full_gt_for_chamfer:
                self.distance_matrix_full = wp.zeros(
                    (self.num_original_points_full, self.num_surface_points), requires_grad=False
                )
                self.neigh_indices_full = wp.zeros(
                    (self.num_original_points_full), dtype=wp.int32, requires_grad=False
                )

        # Parameter to be optimized
        self.wp_spring_Y = wp.from_torch(
            torch.log(torch.tensor(spring_Y, dtype=torch.float32, device=self.device))
            * torch.ones(self.n_springs, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            torch.tensor(
                [collide_object_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            torch.tensor(
                [collide_object_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )

        # --- 新增：全局唯一的排序草稿纸，所有 substep 共用这一块显存 ---
        self.wp_sort_buffer = wp.zeros(
            (self.wp_init_vertices.shape[0], 500), 
            dtype=wp.vec3, 
            requires_grad=False
        )

        # Create the CUDA graph to acclerate
        if cfg.use_graph:
            if cfg.data_type == "real":
                if not disable_backward:
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_loss()
                self.graph = capture.graph
            elif cfg.data_type == "synthetic":
                if not disable_backward:
                    # For synthetic data, we compute simple loss
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_simple_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_simple_loss()
                self.graph = capture.graph
            else:
                raise NotImplementedError

            with wp.ScopedCapture() as forward_capture:
                self.step()
            self.forward_graph = forward_capture.graph
        else:
            self.tape = wp.Tape()

    def set_controller_target(self, frame_idx, pure_inference=False):
        if self.controller_points is not None:
            # Set the controller points
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx - 1]],
                outputs=[self.wp_original_control_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx]],
                outputs=[self.wp_target_control_point],
            )

        if not pure_inference:
            # Set the target points for track loss (downsampled GT)
            wp.launch(
                copy_vec3,
                dim=self.num_original_points,
                inputs=[self.gt_object_points[frame_idx]],
                outputs=[self.wp_current_object_points],
            )

            # 如果提供了 full GT，也更新用于 chamfer loss 的 full GT
            if self.use_full_gt_for_chamfer:
                wp.launch(
                    copy_vec3,
                    dim=self.num_original_points_full,
                    inputs=[self.gt_object_points_full[frame_idx]],
                    outputs=[self.wp_current_object_points_full],
                )

            if cfg.data_type == "real":
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_visibilities[frame_idx]],
                    outputs=[self.wp_current_object_visibilities],
                )
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_motions_valid[frame_idx - 1]],
                    outputs=[self.wp_current_object_motions_valid],
                )

                self.num_valid_visibilities = int(
                    self.gt_object_visibilities[frame_idx].sum()
                )
                self.num_valid_motions = int(
                    self.gt_object_motions_valid[frame_idx - 1].sum()
                )
                
                # 更新 full GT 的 visibility
                if self.use_full_gt_for_chamfer:
                    wp.launch(
                        copy_int,
                        dim=self.num_original_points_full,
                        inputs=[self.gt_object_visibilities_full[frame_idx]],
                        outputs=[self.wp_current_object_visibilities_full],
                    )
                    self.num_valid_visibilities_full = int(
                        self.gt_object_visibilities_full[frame_idx].sum()
                    )

    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v, pure_inference=False):
        # Detach and clone and set requires_grad=True
        assert (
            self.num_object_points == wp_x.shape[0]
            and self.num_object_points == self.wp_states[0].wp_x.shape[0]
        )

        if not pure_inference:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_x, requires_grad=False)],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_v, requires_grad=False)],
                outputs=[self.wp_states[0].wp_v],
            )
        else:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_x],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_v],
                outputs=[self.wp_states[0].wp_v],
            )

    def set_acc_count(self, acc_count):
        if acc_count:
            input = 1
        else:
            input = 0
        wp.launch(
            set_int,
            dim=1,
            inputs=[input],
            outputs=[self.acc_count],
        )

    def update_acc(self):
        wp.launch(
            update_acc,
            dim=self.num_object_points,
            inputs=[
                wp.clone(self.wp_states[0].wp_v, requires_grad=False),
                wp.clone(self.wp_states[-1].wp_v, requires_grad=False),
            ],
            outputs=[self.prev_acc],
        )

    def update_collision_graph(self):
        assert self.object_collision_flag
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_x,
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()
            if not self.controller_points is None:
                # Set the control point
                wp.launch(
                    set_control_points,
                    dim=self.num_control_points,
                    inputs=[
                        self.num_substeps,
                        self.wp_original_control_point,
                        self.wp_target_control_point,
                        i,
                    ],
                    outputs=[self.wp_states[i].wp_control_x],
                )

            # Calculate the spring forces
            wp.launch(
                kernel=eval_springs,
                dim=self.n_springs,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_control_x,
                    self.wp_states[i].wp_control_v,
                    self.num_object_points,
                    self.wp_springs,
                    self.wp_rest_lengths,
                    self.wp_spring_Y,
                    self.wp_dashpot_damping,
                    self.spring_Y_min,
                    self.spring_Y_max,
                ],
                outputs=[self.wp_states[i].wp_vertice_forces],
            )

            if self.object_collision_flag:
                output_v = self.wp_states[i].wp_v_before_collision
            else:
                output_v = self.wp_states[i].wp_v_before_ground

            # Update the output_v using the vertive_forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_vertice_forces,
                    self.wp_masses,
                    self.dt,
                    self.wp_drag_damping,
                    self.reverse_factor,
                ],
                outputs=[output_v],
            )

            if self.object_collision_flag:
                # Update the wp_v_before_ground based on the collision handling
                wp.launch(
                    kernel=object_collision,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_collision,
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                        # --- 传入当前状态记录器 ---
                        # self.wp_states[i].wp_hit_count,
                        # self.wp_states[i].wp_hit_indices,
                        # self.wp_states[i].wp_hit_impulses,

                        self.wp_sort_buffer,  # <--- 修改这里：传入全局复用的草稿纸
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            # Update the x and v
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.wp_collide_fric,
                    self.dt,
                    self.reverse_factor,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )

    def calculate_loss(self):
        # 如果提供了 full GT，则使用 full GT 计算 chamfer loss，否则使用下采样的 GT
        if self.use_full_gt_for_chamfer:
            # 使用未下采样的真值计算 chamfer loss
            wp.launch(
                compute_distances,
                dim=(self.num_original_points_full, self.num_surface_points),
                inputs=[
                    self.wp_states[-1].wp_x,
                    self.wp_current_object_points_full,
                    self.wp_current_object_visibilities_full,
                ],
                outputs=[self.distance_matrix_full],
            )

            wp.launch(
                compute_neigh_indices,
                dim=self.num_original_points_full,
                inputs=[self.distance_matrix_full],
                outputs=[self.neigh_indices_full],
            )

            wp.launch(
                compute_chamfer_loss,
                dim=self.num_original_points_full,
                inputs=[
                    self.wp_states[-1].wp_x,
                    self.wp_current_object_points_full,
                    self.wp_current_object_visibilities_full,
                    self.num_valid_visibilities_full,
                    self.neigh_indices_full,
                    cfg.chamfer_weight,
                ],
                outputs=[self.chamfer_loss],
            )
        else:
            # 使用下采样后的真值计算 chamfer loss
            wp.launch(
                compute_distances,
                dim=(self.num_original_points, self.num_surface_points),
                inputs=[
                    self.wp_states[-1].wp_x,
                    self.wp_current_object_points,
                    self.wp_current_object_visibilities,
                ],
                outputs=[self.distance_matrix],
            )

            wp.launch(
                compute_neigh_indices,
                dim=self.num_original_points,
                inputs=[self.distance_matrix],
                outputs=[self.neigh_indices],
            )

            wp.launch(
                compute_chamfer_loss,
                dim=self.num_original_points,
                inputs=[
                    self.wp_states[-1].wp_x,
                    self.wp_current_object_points,
                    self.wp_current_object_visibilities,
                    self.num_valid_visibilities,
                    self.neigh_indices,
                    cfg.chamfer_weight,
                ],
                outputs=[self.chamfer_loss],
            )

        # Compute the tracking loss - 始终使用下采样后的真值
        wp.launch(
            compute_track_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_motions_valid,
                self.num_valid_motions,
                cfg.track_weight,
            ],
            outputs=[self.track_loss],
        )

        wp.launch(
            compute_acc_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_v,
                self.wp_states[-1].wp_v,
                self.prev_acc,
                self.num_object_points,
                self.acc_count,
                cfg.acc_weight,
            ],
            outputs=[self.acc_loss],
        )

        wp.launch(
            compute_final_loss,
            dim=1,
            inputs=[self.chamfer_loss, self.track_loss, self.acc_loss],
            outputs=[self.loss],
        )

    def calculate_simple_loss(self):
        wp.launch(
            compute_simple_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.num_object_points,
            ],
            outputs=[self.loss],
        )

    def clear_loss(self):
        if cfg.data_type == "real":
            self.distance_matrix.zero_()
            self.neigh_indices.zero_()
            # 如果使用了 full GT，清理对应的 buffer
            if self.use_full_gt_for_chamfer:
                self.distance_matrix_full.zero_()
                self.neigh_indices_full.zero_()
            self.chamfer_loss.zero_()
            self.track_loss.zero_()
            self.acc_loss.zero_()
        self.loss.zero_()

    # Functions used to load the parmeters
    def set_spring_Y(self, spring_Y):
        # assert spring_Y.shape[0] == self.n_springs
        wp.launch(
            copy_float,
            dim=self.n_springs,
            inputs=[spring_Y],
            outputs=[self.wp_spring_Y],
        )

    def set_drag_damping(self, drag_damping):
        """Set drag damping parameter from Warp array"""
        wp.launch(
            copy_float,
            dim=1,
            inputs=[drag_damping],
            outputs=[self.wp_drag_damping],
        )

    def set_dashpot_damping(self, dashpot_damping):
        """Set dashpot damping parameter from Warp array"""
        wp.launch(
            copy_float,
            dim=1,
            inputs=[dashpot_damping],
            outputs=[self.wp_dashpot_damping],
        )

    def set_collide(self, collide_elas, collide_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collision_elas(self, collide_elas):
        """Set collision elasticity parameter from Warp array"""
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )

    def set_collision_fric(self, collide_fric):
        """Set collision friction parameter from Warp array"""
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collide_object(self, collide_object_elas, collide_object_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_elas],
            outputs=[self.wp_collide_object_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_fric],
            outputs=[self.wp_collide_object_fric],
        )

    def set_collision_object_elas(self, collide_object_elas):
        """Set object collision elasticity parameter from Warp array"""
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_elas],
            outputs=[self.wp_collide_object_elas],
        )

    def set_collision_object_fric(self, collide_object_fric):
        """Set object collision friction parameter from Warp array"""
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_fric],
            outputs=[self.wp_collide_object_fric],
        )

    def export_forces_to_txt(self, frame_idx, filename="particle_exact_collision_log.txt", max_points=100):
        """
        导出碰撞日志（基于 Kernel 内的精准记录）。
        """
        if not self.object_collision_flag:
            return

        # 确保 GPU 执行完成
        # wp.synchronize_device(self.device)
        
        masses = self.wp_masses.numpy()
        masks = self.wp_masks.numpy()
        record_limit = min(self.num_object_points, max_points)

        max_record = 5
        first_hit_node = None
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"========== Frame {frame_idx} ==========\n")

            for i in range(self.num_substeps):
                if (self.wp_states[i].wp_hit_count.numpy().max() > 2) and (max_record  > 0):
                    if first_hit_node is None:
                        first_hit_node = self.wp_states[i].wp_hit_count.numpy().argmax()
                    max_record -= 1
                    # 直接拉取 Kernel 里记录的精准碰撞数据
                    hit_count = self.wp_states[i].wp_hit_count.numpy()
                    hit_indices = self.wp_states[i].wp_hit_indices.numpy()
                    
                    hit_impulses = self.wp_states[i].wp_hit_impulses.numpy()
                    pos = self.wp_states[i].wp_x.numpy()

                    # --- 新增：拉取 wp_v_before_ground 数据 ---
                    v_bg = self.wp_states[i].wp_v_before_ground.numpy()
                    
                    substep_logs = []

                    # for p in range(record_limit):
                    p = first_hit_node
                    count = hit_count[p]
                    p_pos = pos[p]
                    p_mask = masks[p]
                    m1 = masses[p]
                    p_v_bg = v_bg[p]  # 获取当前粒子的 v_before_ground
                    # 1. 首先记录当前粒子的基础状态信息
                    log_str = (f"    Particle {p:04d} [Mask:{p_mask}] | "
                                f"Pos: ({p_pos[0]:6.3f}, {p_pos[1]:6.3f}, {p_pos[2]:6.3f}) | "
                                f"V_before_ground: ({p_v_bg[0]:8.4f}, {p_v_bg[1]:8.4f}, {p_v_bg[2]:8.4f})\n")
                    substep_logs.append(log_str)

                    for k in range(count):
                        target_id = hit_indices[p, k]
                        # Kernel 里算的是冲量 J，根据 F = J / (m * dt) 还原出真实的力
                        # 注意：v_new = v - (J_avg / m), 物理学上冲量作用在当前粒子上的效果是 -J / (m * dt)
                        J = hit_impulses[p, k]
                        

                        log_str = (f"    Particle {p:04d} [Mask:{p_mask}] "
                                f"Hit -> Particle {target_id:04d} | "
                                f"Pos: ({p_pos[0]:6.3f}, {p_pos[1]:6.3f}, {p_pos[2]:6.3f}) | "
                                f"Impulse applied: ({J[0]:8.4f}, {J[1]:8.4f}, {J[2]:8.4f}) | \n")
                        substep_logs.append(log_str)
        

                    f.write(f"  --- Substep {i} ---\n")
                    f.writelines(substep_logs)
            f.write("\n")