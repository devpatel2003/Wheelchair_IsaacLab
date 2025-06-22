import omni.replicator.core as rep

class CameraManager:
    def __init__(self, camera_prim_path: str, resolution=(128, 128)):
        self.camera_prim_path = camera_prim_path
        self.resolution = resolution
        self._setup_camera()

    def _setup_camera(self):
        try:
            # Create render product (this connects camera to render pipeline)
            self.render_product = rep.create.render_product(self.camera_prim_path, self.resolution)
            print(f"[CameraManager] Attached camera at {self.camera_prim_path}")
        except Exception as e:
            print(f"[CameraManager] Failed to create render product: {e}")

    def get_rgb(self):
        # Stub for later: add capture logic here
        return None
