from controller.policy.base_image_policy import BaseImagePolicy
from controller.env_runner.base_image_runner import BaseImageRunner

class RealGraspImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir):
        super().__init__(output_dir)
    
    def run(self, policy: BaseImagePolicy):
        return dict()
