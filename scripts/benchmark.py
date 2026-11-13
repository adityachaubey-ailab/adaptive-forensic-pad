import argparse
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.benchmark import BenchmarkRunner

def main():
    parser = argparse.ArgumentParser(description="Run PAD Engine Benchmarks")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the benchmark config")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Device to run on")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"], default="fp32", help="Model precision")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of iterations for latency benchmark")
    parser.add_argument("--output-prefix", type=str, default="benchmark_results", help="Prefix for output report files")
    
    args = parser.parse_args()
    
    config = {
        'model_path': args.model_path,
        'dataset_path': args.dataset_path,
        'device': args.device,
        'precision': args.precision
    }
    
    print("=" * 60)
    print("  Adaptive Forensic PAD Engine — Benchmark Suite")
    print(f"  Model:     {args.model_path}")
    print(f"  Precision: {args.precision.upper()}")
    print(f"  Device:    {args.device.upper()}")
    print("=" * 60)
    
    # Initialize mock model & dataloader for benchmark runner
    model = None
    dataloader = []
    
    runner = BenchmarkRunner(model, dataloader, config)
    
    print("\n[1/3] Running accuracy benchmarks...")
    runner.run_accuracy_benchmark()
    
    print("[2/3] Running latency benchmarks...")
    runner.run_latency_benchmark(n_iterations=args.num_iterations)
    
    print("[3/3] Running resource monitoring...")
    runner.run_resource_benchmark()
    
    report = runner.generate_report()
    print("\n" + report)
    
    runner.save_report(args.output_prefix)
    print(f"\n  Saved report to {args.output_prefix}.json and {args.output_prefix}.md")

if __name__ == "__main__":
    main()
