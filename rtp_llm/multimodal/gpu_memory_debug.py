"""
GPU Memory Debugging Tool for VIT Server

This tool helps diagnose GPU memory leaks by tracking tensor allocations.
Usage:
    import rtp_llm.multimodal.gpu_memory_debug as gpu_debug
    gpu_debug.print_gpu_memory_summary()
    gpu_debug.print_tensor_summary()
"""

import gc
from collections import defaultdict
from typing import Dict, List

import torch


def get_tensor_summary() -> Dict[str, List[Dict]]:
    """
    Get a summary of all tensors in GPU memory.

    Returns:
        Dictionary with device as key, list of tensor info as value.
    """
    tensor_info = defaultdict(list)

    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                if obj.is_cuda:
                    device_str = str(obj.device)
                    tensor_info[device_str].append(
                        {
                            "shape": tuple(obj.shape),
                            "dtype": str(obj.dtype),
                            "size_mb": obj.element_size()
                            * obj.nelement()
                            / (1024 * 1024),
                            "requires_grad": obj.requires_grad,
                            "is_leaf": obj.is_leaf,
                            "grad_fn": str(obj.grad_fn) if obj.grad_fn else None,
                        }
                    )
        except Exception:
            pass

    return dict(tensor_info)


def print_tensor_summary(device: int = 0, top_k: int = 20):
    """
    Print summary of tensors on a specific GPU device.

    Args:
        device: GPU device index
        top_k: Number of largest tensors to show
    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return

    device_str = f"cuda:{device}"
    tensor_info = get_tensor_summary()

    if device_str not in tensor_info:
        print(f"No tensors found on {device_str}")
        return

    tensors = tensor_info[device_str]
    tensors.sort(key=lambda x: x["size_mb"], reverse=True)

    print(f"\n=== GPU Memory Summary for {device_str} ===")
    print(f"Total tensors: {len(tensors)}")
    total_mb = sum(t["size_mb"] for t in tensors)
    print(f"Total tensor memory: {total_mb:.2f} MB")
    print(f"\nTop {min(top_k, len(tensors))} largest tensors:")
    print("-" * 100)
    print(
        f"{'Size (MB)':<12} {'Shape':<30} {'Dtype':<12} {'RequiresGrad':<12} {'GradFn':<30}"
    )
    print("-" * 100)

    for i, t in enumerate(tensors[:top_k]):
        grad_fn_str = str(t["grad_fn"])[:28] if t["grad_fn"] else "None"
        print(
            f"{t['size_mb']:>10.2f}  {str(t['shape']):<30} {t['dtype']:<12} "
            f"{str(t['requires_grad']):<12} {grad_fn_str:<30}"
        )


def print_gpu_memory_summary():
    """Print summary of GPU memory usage using torch.cuda."""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return

    print("\n=== GPU Memory Summary (torch.cuda) ===")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
        print(f"GPU {i}:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Max Allocated: {max_allocated:.2f} GB")
        print()


def get_cache_memory_usage():
    """
    Get memory usage of VIT embedding cache.

    Returns:
        Dictionary with cache statistics.
    """
    from rtp_llm.multimodal.multimodal_util import vit_emb_cache_

    cache_info = {
        "cache_size": 0,
        "total_items": 0,
        "tensor_count": 0,
        "estimated_memory_mb": 0.0,
    }

    if vit_emb_cache_.mm_data_cache is None:
        return cache_info

    cache_info["total_items"] = vit_emb_cache_.mm_data_cache.len()

    tensor_count = 0
    total_memory = 0.0

    with vit_emb_cache_.cache_lock:
        for key, value in vit_emb_cache_.mm_data_cache.items():
            if isinstance(value, (tuple, list)):
                for item in value:
                    if isinstance(item, torch.Tensor) and item.is_cuda:
                        tensor_count += 1
                        total_memory += (
                            item.element_size() * item.nelement() / (1024 * 1024)
                        )
            elif isinstance(value, torch.Tensor) and value.is_cuda:
                tensor_count += 1
                total_memory += value.element_size() * value.nelement() / (1024 * 1024)

    cache_info["tensor_count"] = tensor_count
    cache_info["estimated_memory_mb"] = total_memory

    return cache_info


def print_cache_memory_usage():
    """Print memory usage of VIT embedding cache."""
    cache_info = get_cache_memory_usage()
    print("\n=== VIT Embedding Cache Memory Usage ===")
    print(f"Total cached items: {cache_info['total_items']}")
    print(f"Cached tensors: {cache_info['tensor_count']}")
    print(f"Estimated memory: {cache_info['estimated_memory_mb']:.2f} MB")
    print()


def diagnose_memory_leak():
    """Run a comprehensive memory leak diagnosis."""
    print("=" * 100)
    print("GPU Memory Leak Diagnosis")
    print("=" * 100)

    print_gpu_memory_summary()
    print_cache_memory_usage()

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print_tensor_summary(device=i, top_k=20)


if __name__ == "__main__":
    diagnose_memory_leak()

    # 或者单独查看
    print_gpu_memory_summary()  # GPU显存摘要
    print_cache_memory_usage()  # 缓存使用情况
    print_tensor_summary(device=0, top_k=20)  # 最大的20个tensor
