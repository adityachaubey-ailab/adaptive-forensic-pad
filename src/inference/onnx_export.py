import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Tuple, Any
import time

logger = logging.getLogger(__name__)

def export_to_onnx(model: nn.Module, output_path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224), opset_version: int = 13):
    """
    Export a PyTorch model to ONNX using legacy backend (dynamo=False).
    """
    model.eval()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    dummy_input = torch.randn(*input_shape, device=device)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    logger.info(f"Model exported to {output_path}")

def optimize_onnx(onnx_path: str, output_path: str):
    """
    Optimize ONNX graph using ONNX shape inference.
    """
    import onnx
    import onnx.shape_inference
    
    try:
        model = onnx.load(onnx_path)
        inferred_model = onnx.shape_inference.infer_shapes(model)
        onnx.save(inferred_model, output_path)
        logger.info(f"Shape-inferred ONNX model saved to {output_path}")
    except Exception as e:
        logger.warning(f"ONNX optimization skipped ({e}). Copying original model.")
        import shutil
        shutil.copy(onnx_path, output_path)

def quantize_onnx(onnx_path: str, output_path: str, quant_type: str = 'int8'):
    """
    Dynamic quantization for ONNX model to INT8.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    qtype = QuantType.QInt8 if quant_type == 'int8' else QuantType.QUInt8
        
    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=qtype
    )
    logger.info(f"Quantized INT8 ONNX model saved to {output_path}")

def benchmark_onnx(onnx_path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224), n_iterations: int = 100):
    """
    Benchmark ONNX Runtime execution latency.
    """
    import onnxruntime as ort
    import numpy as np
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
        
    latencies = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    mean_lat = float(np.mean(latencies))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    logger.info(f"ONNX Model Benchmark ({onnx_path}): Mean Latency = {mean_lat:.2f} ms | Throughput = {fps:.1f} FPS")
    return mean_lat, fps

def compare_precision(pytorch_model: Any, onnx_fp32_path: str, onnx_int8_path: str, dataloader: Any = None):
    """
    Compare PyTorch vs ONNX FP32 vs ONNX INT8 latency and parameters.
    """
    input_shape = (1, 3, 224, 224)
    logger.info("--- PRECISION COMPARISON BENCHMARK ---")
    fp32_lat, fp32_fps = benchmark_onnx(onnx_fp32_path, input_shape) if Path(onnx_fp32_path).exists() else (0.0, 0.0)
    int8_lat, int8_fps = benchmark_onnx(onnx_int8_path, input_shape) if Path(onnx_int8_path).exists() else (0.0, 0.0)
    
    print("\n" + "=" * 55)
    print("  ONNX RUNTIME PRECISION COMPARISON TABLE")
    print("=" * 55)
    print(f"  {'Precision':<12} {'Mean Latency':<15} {'Throughput':<15}")
    print("-" * 55)
    print(f"  {'ONNX FP32':<12} {fp32_lat:7.2f} ms     {fp32_fps:7.1f} FPS")
    print(f"  {'ONNX INT8':<12} {int8_lat:7.2f} ms     {int8_fps:7.1f} FPS")
    print("=" * 55 + "\n")

