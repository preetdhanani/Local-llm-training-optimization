"""Local metrics logging utility for dashboard plotting."""
import json
import os
import subprocess
from transformers import TrainerCallback


def get_gpu_info():
    """Queries GPU metrics from PyTorch, NVML, or nvidia-smi."""
    info = {}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu/allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            info["gpu/reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
            info["gpu/max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    except Exception:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Power usage in Watts
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        info["gpu/power_w"] = round(power / 1000.0, 2)
        
        # Temperature in Celsius
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        info["gpu/temp_c"] = temp
        
        # Clocks in MHz
        clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        info["gpu/clock_mhz"] = clock
        
        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info["gpu/total_mb"] = round(mem_info.total / (1024 * 1024), 2)
        info["gpu/used_mb"] = round(mem_info.used / (1024 * 1024), 2)
        info["gpu/free_mb"] = round(mem_info.free / (1024 * 1024), 2)
        
        pynvml.nvmlShutdown()
    except Exception:
        # Fallback to nvidia-smi command
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.current.graphics,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            ).strip()
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 6:
                info["gpu/power_w"] = float(parts[0])
                info["gpu/temp_c"] = int(parts[1])
                info["gpu/clock_mhz"] = int(parts[2])
                info["gpu/total_mb"] = float(parts[3])
                info["gpu/used_mb"] = float(parts[4])
                info["gpu/free_mb"] = float(parts[5])
        except Exception:
            pass
            
    return info


class LocalMetricsCallback(TrainerCallback):
    """
    Hugging Face TrainerCallback that logs SFT or DPO metrics locally to a JSON file.
    
    The resulting JSON file is saved in logs/metrics_{timestamp}.json with the format:
    {
        "sft": [
            { "step": 10, "epoch": 0.1, "loss": 1.23, "eval_loss": 1.45, ... }
        ],
        "dpo": [
            { "step": 10, "epoch": 0.1, "loss": 0.52, "rewards/accuracies": 0.85, ... }
        ]
    }
    """
    def __init__(self, metrics_filepath: str, phase: str):
        """
        Args:
            metrics_filepath: Absolute or relative path to target JSON file.
            phase: Either "sft" or "dpo".
        """
        self.metrics_filepath = os.path.abspath(metrics_filepath)
        self.phase = phase.lower()
        
        # Ensure target folder exists
        os.makedirs(os.path.dirname(self.metrics_filepath), exist_ok=True)
        
        # Initialize file if it does not exist
        if not os.path.exists(self.metrics_filepath):
            with open(self.metrics_filepath, "w", encoding="utf-8") as f:
                json.dump({"sft": [], "dpo": []}, f, indent=2)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Triggered on every training log step."""
        if not logs:
            return

        # Extract only JSON-serializable numeric metrics
        metric_entry = {
            "step": state.global_step,
            "epoch": round(state.epoch, 3) if state.epoch is not None else None,
        }
        
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                # Convert common keys to standard format
                metric_entry[k] = v

        # Append GPU statistics
        try:
            gpu_data = get_gpu_info()
            metric_entry.update(gpu_data)
        except Exception:
            pass

        try:
            # Thread-safe read and append
            data = {"sft": [], "dpo": []}
            if os.path.exists(self.metrics_filepath):
                try:
                    with open(self.metrics_filepath, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            data.update(loaded)
                except Exception:
                    # Fallback if file is corrupted/empty
                    pass
            
            # Check if this metric entry is already recorded for this step
            # (Trainer can call on_log multiple times with the same step for eval vs train)
            history = data.setdefault(self.phase, [])
            
            # Merge if entry exists for this step
            existing = None
            for item in history:
                if item.get("step") == metric_entry["step"]:
                    existing = item
                    break
            
            if existing is not None:
                existing.update(metric_entry)
            else:
                history.append(metric_entry)

            # Save back to file
            with open(self.metrics_filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            # We don't want to crash training due to a logging write error
            import sys
            sys.stderr.write(f"[WARNING] LocalMetricsCallback failed to write: {e}\n")
            sys.stderr.flush()
