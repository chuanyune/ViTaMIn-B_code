# Workspace module for diffusion policy

# Import only the essential workspace classes to avoid dependency issues
try:
    from .train_diffusion_pointnet_workspace import TrainDiffusionPointNetWorkspace
except ImportError:
    pass

try:
    from .train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
except ImportError:
    pass

# Only add to __all__ if import was successful
__all__ = []
if 'TrainDiffusionPointNetWorkspace' in locals():
    __all__.append('TrainDiffusionPointNetWorkspace')
if 'TrainDiffusionUnetImageWorkspace' in locals():
    __all__.append('TrainDiffusionUnetImageWorkspace')
