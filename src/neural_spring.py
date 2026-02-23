import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)
from qqtt.utils import logger, cfg

class FourierFeatureTransform(nn.Module):
    """
    Applies Fourier feature transformation to the input features.
    """
    def __init__(self, input_dim, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        # Output dimension: original features + sin/cos for each frequency
        self.out_dim = input_dim * (2 * num_freqs + 1)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands.to(x.device):
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)

class NeuralSpringField(nn.Module):
    def __init__(self, num_springs, s0_init: torch.Tensor, num_freqs=6):
        """
        Args:
            num_springs: The total number of springs |E| to compute the adaptive spatial resolution N.
            s0_init: (Tensor) The homogeneous physical parameters obtained from the piecewise topology stage. 
                     Shape should be (output_dim,).
            num_freqs: Number of frequencies for the Fourier feature transform.
        """
        super().__init__()
        
        output_dim = 1
 
        # Hyperparameters specified in the paper
        self.C = 32  # Feature channels
        self.N = int(0.85 * math.sqrt(num_springs))  # Adaptive spatial resolution N = 0.85 * sqrt(|E|)
        
        # Tri-plane representations (3 planes: xy, yz, xz)
        self.plane_xy = nn.Parameter(torch.randn(1, self.C, self.N, self.N))
        self.plane_yz = nn.Parameter(torch.randn(1, self.C, self.N, self.N))
        self.plane_xz = nn.Parameter(torch.randn(1, self.C, self.N, self.N))

        for p in [self.plane_xy, self.plane_yz, self.plane_xz]:
            for c in range(self.C):
                # orthogonal_ 需要二维输入，所以我们对 p[0, c] 进行操作
                nn.init.orthogonal_(p[0, c])
        
        # Fourier feature transform module
        self.fourier_transform = FourierFeatureTransform(input_dim=self.C, num_freqs=num_freqs)
        
        # 3-layer MLP with 128 hidden units
        mlp_input_dim = self.fourier_transform.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Initialize S_0 using the provided tensor from the piecewise topology solution
        # It is set as a learnable parameter here so it can be further fine-tuned if needed.
        # If you want to keep it strictly fixed, you can use self.register_buffer('S_0', s0_init.clone())
        self.S_0 = nn.Parameter(s0_init.clone())

    def kaiming_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_warp_simulator(
            self,
            dt,
            init_vertices,
            init_springs,
            init_spring_Y,
            init_rest_lengths,
            init_masses,
            num_object_springs,
            init_masks,
            init_velocities,
            num_all_points,
            num_surface_points,
            num_original_points,
            controller_points,
            object_points,
            object_visibilities,
            object_motions_valid,
            collide_elas,
            collide_fric,
            dashpot_damping,
            drag_damping,
            collide_object_elas,
            collide_object_fric,
            collision_dist,
            reverse_z,
            spring_Y_min,
            spring_Y_max,
            self_collision,
            num_substeps = 5,
            device=None,
    ):
        if hasattr(self, "simulator"):
            self.simulator = None
        
        self.init_vertices = torch.FloatTensor(init_vertices).contiguous().to(device)
        self.init_springs = torch.tensor(init_springs.T, dtype=torch.int32).contiguous().to(device)
        self.init_rest_lengths = torch.FloatTensor(init_rest_lengths).contiguous().to(device)
        self.init_masses = torch.FloatTensor(init_masses).contiguous().to(device)
        self.init_spring_Y = init_spring_Y
        self.object_visibilities = torch.FloatTensor(object_visibilities).contiguous().to(device)
        self.object_motions_valid = torch.FloatTensor(object_motions_valid).contiguous().to(device)
        self.init_velocities = torch.FloatTensor(init_velocities).contiguous().to(device)
        self.controller_points = torch.FloatTensor(controller_points).contiguous().to(device)\
              if controller_points is not None else None
        self.object_points = torch.FloatTensor(object_points).contiguous().to(device)

        self.num_object_springs = num_object_springs
        self.dt = dt
        self.collide_elas = collide_elas
        self.collide_fric = collide_fric
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.collide_object_elas = collide_object_elas
        self.collide_object_fric = collide_object_fric
        self.collision_dist = collision_dist
        self.reverse_z = reverse_z
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max
        self.self_collision = self_collision
        self.num_all_points = num_all_points
        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        self.num_substeps = num_substeps

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=self.dt,
            num_substeps=self.num_substeps,
            spring_Y=self.init_spring_Y, 
            collide_elas=self.collide_elas, 
            collide_fric=self.collide_fric, 
            dashpot_damping=int(self.dashpot_damping), # DEBUG
            drag_damping=int(self.drag_damping), # DEBUG
            collide_object_elas=self.collide_object_elas,
            collide_object_fric=self.collide_object_fric,
            init_masks=init_masks,
            collision_dist=self.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=self.reverse_z,
            spring_Y_min=self.spring_Y_min,
            spring_Y_max=self.spring_Y_max,
            gt_object_points=self.object_points, 
            gt_object_visibilities=self.object_visibilities.bool(),
            gt_object_motions_valid=self.object_motions_valid.bool(),
            self_collision=self.self_collision,
        )

        self.simulator.set_init_state(
                self.simulator.wp_init_vertices,
                self.simulator.wp_init_velocities
        )

    def sample_plane(self, plane, coords):
        """
        Helper function to query tri-plane features via bilinear interpolation.
        Assumes coords are normalized to [-1, 1].
        """
        # Reshape coords for grid_sample: (Batch, 2) -> (1, Batch, 1, 2)
        grid = coords.unsqueeze(0).unsqueeze(2) 
        
        # Sample features: Output shape is (1, C, Batch, 1)
        sampled = F.grid_sample(plane, grid, align_corners=True, padding_mode='border')
        
        # Reshape back to (Batch, C)
        return sampled.squeeze(3).squeeze(0).transpose(0, 1)

    def forward(self, x_mid):
        """
        Forward pass for Neural Spring Fields.
        Args:
            x_mid: (Batch, 3) Endpoints of the springs.
           
        Returns:
            S_e: (Batch, output_dim) The predicted physical properties for each spring.
        """
        
        # Note: x_mid coordinates should be normalized between [-1, 1] for grid_sample to work correctly.
        # Assuming normalization has already been applied to inputs here.
        
        # 2. Project x_mid onto the three orthogonal planes
        xy = x_mid[:, [0, 1]]
        yz = x_mid[:, [1, 2]]
        xz = x_mid[:, [0, 2]]
        
        # 3. Query the corresponding features on each plane by bilinear interpolation
        feat_xy = self.sample_plane(self.plane_xy, xy)
        feat_yz = self.sample_plane(self.plane_yz, yz)
        feat_xz = self.sample_plane(self.plane_xz, xz)
        
        # 4. Aggregate features through the sum operation
        aggregated_feat = feat_xy + feat_yz + feat_xz
        
        # 5. Apply Fourier feature transform
        feat_pe = self.fourier_transform(aggregated_feat)

        # 6. Pass into the 3-layer MLP
        delta_S = self.mlp(feat_pe)
        
        # 7. Add to the initialization parameter S_0
        S_e = self.S_0[:, None] + delta_S * 100
        
        # clip 
        S_e = torch.clamp(S_e, 1e-2, cfg.spring_Y_max)

        return S_e