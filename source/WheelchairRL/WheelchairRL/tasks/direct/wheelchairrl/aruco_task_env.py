import torch
import math
import numpy as np
import cv2
import cv2.aruco as aruco
import omni.usd
from pxr import UsdGeom
from collections.abc import Sequence

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

import omni.usd
from pxr import Usd, UsdGeom, UsdShade, Sdf, Vt, Gf, UsdPhysics
import random
from gymnasium import spaces




class ArucoTaskEnv(DirectRLEnv):
    cfg: ArucoTaskEnvCfg

    def __init__(self, cfg: ArucoTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

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


    
    def _spawn_tag(self, stage, env_index, image_path, seed=None):
        """Spawns a flat textured plane with UVs on one side only (front-facing)."""
        rng = random.Random(seed + env_index if seed is not None else None)

        # Random position within 10×10 area (X, Y), height (Z)
        x = rng.uniform(-5.0, 5.0)
        y = rng.uniform(-5.0, 5.0)
        z = rng.uniform(0.2, 1.1)
        rot_z = rng.uniform(0.0, 360.0)  # Random rotation around Z-axis

        # Paths
        env_path = f"/World/envs/env_{env_index}"
        plane_path = f"{env_path}/aruco_tag"
        material_path = f"/World/Materials/ArucoMaterial_{env_index}"

        # Create plane as a flat cube with 1 face
        plane = UsdGeom.Mesh.Define(stage, plane_path)

        # Vertices of a square plane in X-Y
        points = [
            Gf.Vec3f(-0.25, -0.25, 0.0),
            Gf.Vec3f( 0.25, -0.25, 0.0),
            Gf.Vec3f( 0.25,  0.25, 0.0),
            Gf.Vec3f(-0.25,  0.25, 0.0),
        ]

        # Single quad face
        face_vertex_counts = [4]
        face_vertex_indices = [0, 1, 2, 3]

        # UVs for face vertices
        st = Vt.Vec2fArray([
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(0.0, 1.0),
        ])

        # Assign geometry
        plane.CreatePointsAttr(points)
        plane.CreateFaceVertexCountsAttr(face_vertex_counts)
        plane.CreateFaceVertexIndicesAttr(face_vertex_indices)

        # Assign UVs
        prim = plane.GetPrim()
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        uv_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        uv_primvar.Set(st)
        uv_primvar.SetInterpolation("faceVarying")

        # 2. Create material
        material = UsdShade.Material.Define(stage, material_path)

        # 3. Texture shader
        texture = UsdShade.Shader.Define(stage, material_path + "/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(image_path)
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        # 4. Primvar reader
        reader = UsdShade.Shader.Define(stage, material_path + "/PrimvarReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        # 5. Connect UV → texture
        texture.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(reader.GetOutput("result"))

        # 6. Surface shader
        shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).ConnectToSource(texture.GetOutput("rgb"))
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

        # 7. Bind material
        material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))
        UsdShade.MaterialBindingAPI(prim).Bind(material)

        # 8. Transform in front of robot
        UsdGeom.XformCommonAPI(plane).SetTranslate((x,y,z))
        UsdGeom.XformCommonAPI(plane).SetRotate((0.0, 90.0, rot_z))
        UsdGeom.XformCommonAPI(plane).SetScale((0.6, 0.6, 0.1))  # Width, thickness, height



        print(f"[INFO] Textured plane added at {plane_path}")
    
    def spawn_walls(self, stage, env_index, area_size=10.0, wall_height=1.0, wall_thickness=0.1, wall_z=0.5):
        """Spawn 4 walls around a 10x10 area for a single environment."""
        half_size = area_size / 2.0
        env_path = f"/World/envs/env_{env_index}"

        wall_params = [
            # Front (positive Y)
            ("wall_front", (0.0, half_size + wall_thickness / 2.0, wall_z), (area_size, wall_thickness, wall_height)),
            # Back (negative Y)
            ("wall_back", (0.0, -half_size - wall_thickness / 2.0, wall_z), (area_size, wall_thickness, wall_height)),
            # Right (positive X)
            ("wall_right", (half_size + wall_thickness / 2.0, 0.0, wall_z), (wall_thickness, area_size, wall_height)),
            # Left (negative X)
            ("wall_left", (-half_size - wall_thickness / 2.0, 0.0, wall_z), (wall_thickness, area_size, wall_height)),
        ]

        for name, pos, scale in wall_params:
            wall_path = f"{env_path}/{name}"
            wall = UsdGeom.Cube.Define(stage, wall_path)
            UsdGeom.XformCommonAPI(wall).SetTranslate(pos)
            UsdGeom.XformCommonAPI(wall).SetScale(scale)
            UsdPhysics.CollisionAPI.Apply(wall.GetPrim())


    def randomize_ground_texture(self, stage, texture_paths, env_index=0):
        # Choose one texture randomly
        chosen_texture = random.choice(texture_paths)
        print(f"[INFO] Randomly selected ground texture: {chosen_texture}")

        ground_path = f"/World/envs/env_{env_index}/GroundPlane"
        material_path = f"/World/envs/env_{env_index}/GroundMaterial"
        shader_path = material_path + "/Shader"
        texture_node_path = material_path + "/Texture"

        # Get ground prim
        ground_prim = stage.GetPrimAtPath(ground_path)
        if not ground_prim.IsValid():
            print(f"[WARNING] GroundPlane not found at {ground_path}")
            return

        # ✅ Add UV primvar
        
        primvars_api = UsdGeom.PrimvarsAPI(ground_prim)
        uv_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)

        # Basic UV mapping assuming quad ground mesh
        uvs = Vt.Vec2fArray([
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 1.0),
        ])
        uv_primvar.Set(uvs)
        uv_primvar.SetInterpolation("faceVarying")

        # ✅ Add Primvar Reader
        reader = UsdShade.Shader.Define(stage, material_path + "/PrimvarReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        # Create material + shader
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

        # Texture shader
        texture = UsdShade.Shader.Define(stage, texture_node_path)
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(chosen_texture)
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        texture.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(reader.GetOutput("result"))

        # Connect to shader
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).ConnectToSource(texture.GetOutput("rgb"))
        material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput("surface"))

        # Bind to ground
        UsdShade.MaterialBindingAPI(ground_prim).Bind(material)
        UsdGeom.XformCommonAPI(ground_prim).SetScale((1, 1, 0.1))  # Width, thickness, height

    def _create_uv_ground(self, stage, env_index=0):
        ground_path = f"/World/envs/env_{env_index}/GroundPlane"
        mesh = UsdGeom.Mesh.Define(stage, ground_path)

        # Create quad vertices for a 10x10 plane
        half_size = 5.0
        points = [
            Gf.Vec3f(-half_size, -half_size, 0.0),
            Gf.Vec3f( half_size, -half_size, 0.0),
            Gf.Vec3f( half_size,  half_size, 0.0),
            Gf.Vec3f(-half_size,  half_size, 0.0),
        ]
        indices = [0, 1, 2, 3]
        counts = [4]

        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateFaceVertexCountsAttr(counts)

        # UVs
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        uvs = Vt.Vec2fArray([
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 1.0),
        ])
        uv_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        uv_primvar.Set(uvs)
        uv_primvar.SetInterpolation("faceVarying")

        UsdGeom.XformCommonAPI(mesh).SetTranslate((0.0, 0.0, 0.005))


    def _setup_scene(self):

        self.robot = Articulation(self.cfg.robot_cfg)

        light_cfg = DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

        self.camera = Camera(CameraCfg(
            prim_path=(
                "/World/envs/env_0/Robot/chassis_link/camera_mount/carter_camera_first_person"
            ),
            width=500,
            height=500,
            data_types=["rgb"],
            spawn = None,
            
        ))
        
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())


        ASSET_TAG_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "tag"
        ARUCO_IMAGE_PATH = (ASSET_TAG_ROOT / "aruco_0.png").as_posix()  # Safe for USD
        ASSET_GROUND_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "ground"
        TEXTURE_PATHS = [str(p) for ext in ("*.png", "*.jpg", "*.jpeg") for p in ASSET_GROUND_ROOT.glob(ext)]
        stage = omni.usd.get_context().get_stage()
        for i in range(self.num_envs):
            ground_path = f"/World/envs/env_{i}/GroundPlane"
            ground_cfg = GroundPlaneCfg(size=(10.0, 10.0), visible=True)
            self._spawn_tag(stage, env_index=i, image_path=ARUCO_IMAGE_PATH)
            self.spawn_walls(stage, env_index=i)
            self._create_uv_ground(stage, env_index=i)
            self.randomize_ground_texture(stage, TEXTURE_PATHS, env_index=i)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        #self.camera  = self.scene.sensors["fpv"]          # ← add this line
        print(f"[INFO] started scene setup with {self.num_envs} environments.")



    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        print(f"Actions: {self.actions}")
        #self.actions = torch.tensor([[1.0, 0.0]], device=self.device)  # Fixed action for testing
        



    def _apply_action(self):
        self.robot.write_joint_damping_to_sim(damping=1.0, joint_ids=[self._left_wheel_idx[0], self._right_wheel_idx[0]])

    # Scale actions from [-1, 1] to velocity targets
        scale_v = 5.0 * self.wheel_radius                      # → ~0.33
        scale_w = 10.0 * self.wheel_radius / (self.axle_length / 2.0)  # → ~2.2

        v = scale_v * self.actions[:, 0]  # scaled linear velocity
        w = scale_w * self.actions[:, 1]  # scaled angular velocity

        # Differential drive: linear + angular velocity → wheel linear velocity
        v_l = v - (self.axle_length / 2.0) * w
        v_r = v + (self.axle_length / 2.0) * w

        # Convert to angular velocity (rad/s)
        target_vel_l = v_l / self.wheel_radius
        target_vel_r = v_r / self.wheel_radius
        target_vel = torch.stack([target_vel_l, target_vel_r], dim=1)
        #print(f"[DEBUG] Target velocities: {target_vel}")

        # Get current angular velocities from simulation
        current_vel = self.robot.data.joint_vel[:, [self._left_wheel_idx[0], self._right_wheel_idx[0]]]


        # Clamp torque to physical limits
        max_torque = 10.0  # Set this to your robot’s safe max torque
        torque_cmd = torch.clamp(target_vel, -max_torque, max_torque)
        # Apply the torque commands
        self.robot.set_joint_effort_target(
            target=target_vel,
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
    
    