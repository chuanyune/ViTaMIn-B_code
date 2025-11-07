from typing import Optional, Dict
import os


"""
Common Python map types:
1. Dictionary: key-value
2. map function: key-value
# This is Python's built-in function used to apply a function to each element in an iterable
numbers = [1, 2, 3]
squared = map(lambda x: x*x, numbers)  # Returns map object
"""

class TopKCheckpointManager:
    """
    Manages checkpoint files during model training, keeping only the k best performing checkpoints.
    Supports both max and min modes.
    """
    def __init__(self,
            save_dir,              # Checkpoint save directory
            monitor_key: str,      # Metric name to monitor (e.g. 'train_loss')
            mode='min',           # Mode: 'min' keeps smallest values, 'max' keeps largest values
            k=1,                  # Number of checkpoints to keep
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt' # Checkpoint filename format
        ):
        # Validate parameters
        assert mode in ['max', 'min']
        assert k >= 0

        # Initialize attributes
        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()  # Store mapping between checkpoint paths and metric values
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        """
        Decide whether to save a new checkpoint based on monitored metric.
        
        Args:
            data: Dictionary containing monitored metrics, e.g. {'epoch': 10, 'train_loss': 0.5}
            
        Returns:
            str or None: If checkpoint should be saved, return save path; otherwise return None
        """
        # If k=0, don't save any checkpoints
        if self.k == 0:
            return None

        # Get monitored metric value and checkpoint save path
        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        # If current number of saved checkpoints is less than k, save directly
        if len(self.path_value_map) < self.k:
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # Reached maximum save count k, need to decide whether to replace existing checkpoint
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]     # Get checkpoint with minimum value
        max_path, max_value = sorted_map[-1]    # Get checkpoint with maximum value

        # Decide whether to delete existing checkpoint based on mode
        delete_path = None
        if self.mode == 'max':
            # max mode: when new value is greater than minimum, delete minimum checkpoint
            if value > min_value:
                delete_path = min_path
        else:
            # min mode: when new value is less than maximum, delete maximum checkpoint
            if value < max_value:
                delete_path = max_path

        # If no checkpoint needs to be deleted, return None
        if delete_path is None:
            return None
        else:
            # Update checkpoint mapping
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            # Ensure save directory exists
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            # Delete old checkpoint file
            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
