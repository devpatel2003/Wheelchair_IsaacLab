import torch
import math
import numpy as np
import cv2
import cv2.aruco as aruco
from collections.abc import Sequence
import omni.replicator.core as rep

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim import DomeLightCfg
from isaaclab.utils.math import sample_uniform

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg, save_images_to_file
#from isaaclab.sensors.camera.camera_cfg import CameraCfg
#from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


from pathlib import Path
import torchvision    






from .aruco_task_env_cfg import ArucoTaskEnvCfg
from .camera_manager import CameraManager



class ArucoTaskEnv(DirectRLEnv):
    cfg: ArucoTaskEnvCfg

    def __init__(self, cfg: ArucoTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._target_pos = torch.tensor([5.0, 0.0, 0.0], device=self.device)
        self._collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_range = torch.full((self.num_envs,), 100.0, device=self.device)
        self.prev_pose = None
        self.curr_range = torch.full((self.num_envs,), 100.0, device=self.device)
        self.heading_error  = torch.zeros(self.num_envs,       device=self.device)
        self.seen_last_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)



        self.rep_camera = None
        self.camera_prim_path = "/World/envs/env_0/Robot/carter/chassis_link/camera_mount/carter_camera_first_person"
        self._saved_debug_frame = False


        self.TAG_SIZE = 0.16
        self.K = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]])
        self.D = np.zeros(5)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_obs = torch.zeros((self.num_envs, 4), device=self.device)
        self.seen_last_step = False

        self.wheel_radius = 0.033  # in meters
        self.axle_length = 0.3     # distance between wheels (adjust as needed)
        self.max_speed = 100  # max speed in m/s
        self.max_angular_speed = 1  # max angular speed in rad/s
    

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

        self.camera = Camera(CameraCfg(
            prim_path=(
                "/World/envs/env_0/Robot/chassis_link/camera_mount/carter_camera_third_person"
            ),
            width=500,
            height=500,
            data_types=["rgb"],
            spawn = None,
            
        ))
        
    


        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        #self.camera  = self.scene.sensors["fpv"]          # ← add this line



    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.actions = torch.tensor([[1.0, 0.0]], device=self.device)  # Fixed action for testing
        



    def _apply_action(self):
        self.robot.write_joint_damping_to_sim(damping=1.0, joint_ids=[self._left_wheel_idx[0], self._right_wheel_idx[0]])

    # Scale actions from [-1, 1] to velocity targets
        v = self.max_speed * self.actions[:, 0] # m/s
        w = self.max_angular_speed * self.actions[:, 1]  # rad/s

        # Differential drive: linear + angular velocity → wheel linear velocity
        v_l = v - (self.axle_length / 2.0) * w
        v_r = v + (self.axle_length / 2.0) * w

        # Convert to angular velocity (rad/s)
        target_vel_l = v_l / self.wheel_radius
        target_vel_r = v_r / self.wheel_radius
        target_vel = torch.stack([target_vel_l, target_vel_r], dim=1)

        # Get current angular velocities from simulation
        current_vel = self.robot.data.joint_vel[:, [self._left_wheel_idx[0], self._right_wheel_idx[0]]]
        print(current_vel)


        # Clamp torque to physical limits
        max_torque = 2.0  # Set this to your robot’s safe max torque
        torque_cmd = torch.clamp(target_vel, -max_torque, max_torque)

        # Apply the torque commands
        self.robot.set_joint_effort_target(
            target=torque_cmd,
            joint_ids=[int(self._left_wheel_idx[0]), int(self._right_wheel_idx[0])]
        )
        self.robot.write_data_to_sim()


    def _get_observations(self) -> dict:


        raw_camera_data = self.camera.data.output["rgb"]
        norm_camera_data = raw_camera_data / 255.0 
            # normalize the camera data for better training results
        mean_tensor = torch.mean(norm_camera_data, dim=(1, 2), keepdim=True)
        camera_data = raw_camera_data - mean_tensor
        observations = {"policy": raw_camera_data.clone()}
        #print("[DEBUG] Camera mean pixel:", raw_camera_data.mean().item())

    
        if self.episode_length_buf % 50 == 0:    
            save_images_to_file(norm_camera_data, f"cartpole_rgb.png")
            print("[DEBUG] Saved first RGB frame to disk.")



        # Grab RGB data — since replicator doesn’t return the image to Python,
        # you’ll just save it to disk for now. You can later load it via PIL/OpenCV

        # Return your observation dictionary
        return {
            "rgb": camera_data,
            "aruco": torch.zeros((7,), device=self.device),  # temp
            "odom": torch.zeros((3,), device=self.device),   # temp
            "collision": torch.tensor([0], device=self.device),
        }

    def _get_rewards(self) -> torch.Tensor:
        reward = torch.full((self.num_envs,), -0.01, device=self.device)

        # visibility flag for env-0 as a Python bool
        in_view_flag = (self.aruco_obs[0, 0] == 1).item()

        if in_view_flag and not self.seen_last_step:
            reward += 1.0
            self.seen_last_step = True

        if in_view_flag:
            delta_range = self.prev_range - self.curr_range
            reward += 0.5 * delta_range
            reward += 0.1 * torch.cos(self.heading_error)

            if self.curr_range < 0.15 and abs(self.heading_error) < 0.087:
                reward += 5.0
                self._done = True

        if self._collision.any():
            reward -= 2.0

        self.prev_range = self.curr_range
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dist = torch.norm(self.robot.data.root_pos_w[:, :2] - self._target_pos[:2], dim=-1)
        reached = dist < 0.15
        return reached, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if not hasattr(self, "_left_wheel_idx"):  # First-time setup
            self._left_wheel_idx, _ = self.robot.find_joints(["left_wheel"])
            self._right_wheel_idx, _ = self.robot.find_joints(["right_wheel"])
            print("[DEBUG] Joint names:", self.robot.data.joint_names)

        self.prev_pose = self.robot.data.root_pos_w.clone()

    def _compute_odometry(self):
        curr_pose = self.robot.data.root_pos_w
        if self.prev_pose is None:
            self.prev_pose = curr_pose.clone()
        delta = curr_pose - self.prev_pose
        dist = torch.norm(delta[:, :2], dim=-1)
        yaw_rate = yaw_rate = self.robot.data.root_ang_vel_w[:, 2]
        heading = torch.atan2(delta[:, 1], delta[:, 0])
        self.prev_pose = curr_pose.clone()
        return torch.stack([dist, heading, yaw_rate], dim=-1)

    def _detect_aruco(self, rgb_tensor):
        rgb = rgb_tensor.cpu().numpy().transpose(1, 2, 0)
        corners, ids, _ = aruco.detectMarkers(rgb, self.aruco_dict)

        aruco_obs = np.zeros((7,), dtype=np.float32)
        tag_range = torch.tensor(100.0, device=self.device)
        heading_error = torch.tensor(0.0, device=self.device)

        if ids is not None:
            _, rvec, tvec = aruco.estimatePoseSingleMarkers(corners, self.TAG_SIZE, self.K, self.D)
            tag_cam = tvec[0][0]
            x_px, y_px = corners[0][0][0]
            area = cv2.contourArea(corners[0])
            aruco_obs = np.array([1, x_px, y_px, area, *tag_cam])
            tag_range = torch.tensor(np.linalg.norm(tag_cam), device=self.device)
            heading_error = torch.tensor(math.atan2(tag_cam[0], tag_cam[2]), device=self.device)

        return torch.tensor(aruco_obs, device=self.device), tag_range, heading_error