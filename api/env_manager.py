import sys

def check_environment():
    """
    Perform a pre-flight check of the environment.
    """
    results = {
        "status": "ready",
        "python_version": sys.version,
        "torch_installed": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "bitsandbytes_available": False,
        "errors": []
    }

    try:
        import torch
        results["torch_installed"] = True
        results["cuda_available"] = torch.cuda.is_available()
        results["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if results["cuda_available"]:
            for i in range(results["device_count"]):
                results["devices"].append({
                    "name": torch.cuda.get_device_name(i),
                    "vram": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                })
    except Exception as e:
        results["errors"].append(f"torch loading failed: {e}")
        results["status"] = "warning"
    
    # Check bitsandbytes
    try:
        import bitsandbytes as bnb
        results["bitsandbytes_available"] = True
    except Exception as e:
        results["errors"].append(f"bitsandbytes loading failed: {e}")
        if results["status"] == "ready":
             results["status"] = "warning"

    # Overall status
    if not results["cuda_available"]:
        results["status"] = "warning"
        results["errors"].append("CUDA is not available. Training will be extremely slow on CPU.")

    return results
