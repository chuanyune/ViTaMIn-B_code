import sys
import os
from datetime import datetime


class TeeOutput:
    """Class that outputs to both terminal and log file simultaneously"""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file
    
    def write(self, text):
        # Write to terminal
        self.terminal.write(text)
        self.terminal.flush()
        
        # Write to log file (add timestamp)
        if self.log_file is not None and text.strip():  # Ignore empty lines
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            # Add timestamp to each line
            lines = text.rstrip('\n').split('\n')
            for line in lines:
                if line.strip():  # Only process non-empty lines
                    self.log_file.write(f"[{timestamp}] {line}\n")
                else:
                    self.log_file.write("\n")
            self.log_file.flush()
    
    def flush(self):
        if hasattr(self.terminal, 'flush'):
            self.terminal.flush()
        if self.log_file is not None:
            self.log_file.flush()


def setup_logging(root_dir: str, script_name: str, config_path: str = None):
    """
    Setup logging system, redirect all print output to log file
    
    Args:
        root_dir: Project root directory
        script_name: Script name (used for log file naming)
        config_path: Configuration file path (optional, for recording)
    
    Returns:
        tuple: (log_filepath, original_stdout, log_file) for cleanup
    """
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(root_dir, "logs", script_name.replace('.py', ''))
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{script_name.replace('.py', '')}_{timestamp}.txt"
    log_filepath = os.path.join(log_dir, log_filename)
    log_file = open(log_filepath, 'w', encoding='utf-8')
    
    # Save original stdout and redirect
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, log_file)
    
    # Record start information
    print("=" * 60)
    print(f"=== Starting evaluation script: {script_name} ===")
    if config_path:
        print(f"Config file: {config_path}")
    print(f"Log file: {log_filepath}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return log_filepath, original_stdout, log_file


def cleanup_logging(log_filepath: str, original_stdout, log_file):
    """
    Cleanup logging system, restore original stdout and close log file
    
    Args:
        log_filepath: Log file path
        original_stdout: Original stdout object
        log_file: Log file object
    """
    # Record final information before restoring stdout
    if log_file is not None:
        print("=" * 60)
        print(f"Evaluation script end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=== Evaluation script execution complete ===")
        print("=" * 60)
    
    # Restore original stdout and close log file
    if original_stdout is not None:
        sys.stdout = original_stdout
    
    if log_file is not None:
        log_file.close()
        print(f"Log saved to: {log_filepath}") 