#!/usr/bin/env python
import argparse
import logging
import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pad_model import PADBaselineModel, PADModel
from src.inference.onnx_export import export_to_onnx, optimize_onnx, quantize_onnx, benchmark_onnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument('--model-path', type=str, required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX path')
    parser.add_argument('--quantize', choices=['fp32', 'fp16', 'int8'], default='fp32', help='Quantization type')
    parser.add_argument('--model-type', choices=['baseline', 'full'], default='baseline', help='Model architecture')
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info(f"Exporting model {args.model_path} to {args.output}...")
    
    # Load model architecture
    if args.model_type == 'baseline':
        model = PADBaselineModel(model_type='small', pretrained=False, num_classes=2)
    else:
        model = PADModel(model_type='small', pretrained=False, num_classes=2)
        
    if Path(args.model_path).exists():
        state_dict = torch.load(args.model_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        logger.info(f"Successfully loaded checkpoint: {args.model_path}")
    else:
        logger.warning(f"Checkpoint {args.model_path} not found. Exporting unweighted architecture.")
        
    model.eval()
    input_shape = (1, 3, 224, 224)
    
    # 1. Export to ONNX
    export_to_onnx(model, args.output, input_shape)
    
    # 2. Graph Optimization
    opt_output = args.output.replace('.onnx', '_opt.onnx')
    optimize_onnx(args.output, opt_output)
    final_model = opt_output if Path(opt_output).exists() else args.output
    
    # 3. Quantization
    if args.quantize == 'int8':
        q_output = final_model.replace('.onnx', '_int8.onnx')
        quantize_onnx(final_model, q_output, quant_type='int8')
        if Path(q_output).exists():
            final_model = q_output
            
    # 4. Benchmark ONNX model
    benchmark_onnx(final_model, input_shape)

if __name__ == '__main__':
    main()
