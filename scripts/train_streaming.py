"""
Streaming Trainer for Adaptive Forensic PAD Engine

Streams CelebA-Spoof directly from Hugging Face in real-time,
trains the model in parallel, and discards sample data immediately
from memory without saving GBs of image files to disk!

Usage:
    python scripts/train_streaming.py --model baseline --batch-size 32 --steps 1000
    python scripts/train_streaming.py --model full --batch-size 16 --steps 2000 --lr 1e-4
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pad_model import PADBaselineModel, PADModel
from src.models.full_pipeline import AdaptiveForensicPAD
from src.benchmark.metrics import compute_apcer, compute_bpcer, compute_acer, compute_auc


class HFStreamingPADDataset(IterableDataset):
    """
    Iterable PyTorch Dataset that streams CelebA-Spoof samples live from HuggingFace,
    converts them to tensors in memory, and immediately discards raw sample bytes.
    """
    def __init__(self, hf_repo: str = "Ar4ikov/celebA_spoof", split: str = "train", img_size: int = 224):
        super().__init__()
        self.hf_repo = hf_repo
        self.split = split
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __iter__(self):
        from datasets import load_dataset
        # Load streaming dataset from HF with a light 500-sample shuffle buffer for balanced batches
        dataset = load_dataset(self.hf_repo, streaming=True, split=self.split).shuffle(buffer_size=500, seed=42)
        
        for sample in dataset:
            try:
                # 1. Extract Image
                img = None
                for key in ["image", "img", "pixel_values", "photo", "Filepath"]:
                    if key in sample and sample[key] is not None:
                        img = sample[key]
                        break
                
                if img is None:
                    continue
                
                # If img is a file path or string
                if isinstance(img, str):
                    if os.path.exists(img):
                        img = Image.open(img)
                    else:
                        continue
                elif not isinstance(img, Image.Image):
                    try:
                        img = Image.fromarray(np.array(img))
                    except Exception:
                        continue
                
                img_tensor = self.transform(img.convert("RGB"))
                
                # 2. Extract Label (0 = bona_fide/real, 1 = spoof/attack)
                raw_label = None
                for key in ["Class", "class", "spoof_label", "is_spoof", "label", "labels"]:
                    if key in sample and sample[key] is not None:
                        raw_label = sample[key]
                        break
                
                if raw_label is None:
                    continue
                
                if isinstance(raw_label, (list, tuple)):
                    raw_label = raw_label[0] if len(raw_label) > 0 else "live"
                
                if isinstance(raw_label, str):
                    label = 0 if raw_label.lower().strip() in ["live", "real", "bona_fide", "0"] else 1
                else:
                    label = 0 if int(raw_label) == 0 else 1
                
                # Yield tensor and label directly from RAM
                yield img_tensor, label
                
            except Exception:
                continue


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming Train for PAD Engine")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "full", "adaptive"],
                        help="Model architecture to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps/batches")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hf-repo", type=str, default="Ar4ikov/celebA_spoof", help="HF dataset repo")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print("=" * 60)
    print("  Adaptive Forensic PAD Engine — Real-time Streaming Trainer")
    print(f"  Dataset: {args.hf_repo} (Streaming Mode)")
    print(f"  Model:   {args.model.upper()}")
    print(f"  Device:  {device}")
    print("  Memory:  Streaming into RAM -> Training -> Auto-delete")
    print("=" * 60)

    # 1. Initialize Dataset & Dataloader
    print("\n[1/3] Connecting to Hugging Face stream...")
    stream_dataset = HFStreamingPADDataset(hf_repo=args.hf_repo, split="train")
    dataloader = DataLoader(stream_dataset, batch_size=args.batch_size)

    # 2. Initialize Model
    print(f"[2/3] Building model '{args.model}'...")
    if args.model == "baseline":
        model = PADBaselineModel(model_type="small", pretrained=True, num_classes=2).to(device)
    elif args.model == "full":
        model = PADModel(model_type="small", pretrained=True, num_classes=2).to(device)
    else:
        model = AdaptiveForensicPAD(model_type="small", pretrained=True, num_classes=2).to(device)

    print(f"       Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 3. Training Loop
    print("\n[3/3] Starting parallel stream & train loop...\n")
    model.train()
    
    start_time = time.time()
    running_loss = 0.0
    all_y_true = []
    all_y_pred = []
    all_y_scores = []
    
    for step, (images, labels) in enumerate(dataloader, 1):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if args.model == "baseline":
            outputs = model(images)
            loss = criterion(outputs, labels)
        elif args.model == "full":
            head_dict = model(images)
            # Combine texture & frequency loss
            loss = criterion(head_dict["texture_score"], labels) + criterion(head_dict["frequency_score"], labels)
            outputs = head_dict["texture_score"]
        else:
            outputs, _ = model.forward_fast(images)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_y_true.extend(labels.cpu().numpy())
        all_y_pred.extend(preds.cpu().numpy())
        all_y_scores.extend(probs[:, 1].detach().cpu().numpy())
        
        if step % 10 == 0 or step == args.steps:
            avg_loss = running_loss / (10 if step % 10 == 0 else step % 10)
            running_loss = 0.0
            elapsed = time.time() - start_time
            img_per_sec = (step * args.batch_size) / elapsed
            print(f"  Step [{step:4d}/{args.steps}] | Loss: {avg_loss:.4f} | Speed: {img_per_sec:.1f} img/s | Time: {elapsed:.1f}s")
        
        if step >= args.steps:
            break

    # Final Metrics
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETED & DATA AUTOMATICALLY DISCARDED")
    print("=" * 60)
    
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_scores = np.array(all_y_scores)
    
    bpcer = compute_bpcer(y_true, y_pred)
    auc = compute_auc(y_true, y_scores)
    acc = np.mean(y_true == y_pred) * 100
    
    print(f"  Processed Images: {len(y_true):,}")
    print(f"  Accuracy:         {acc:.2f}%")
    print(f"  BPCER (FRR):      {bpcer:.4f}")
    print(f"  AUC:              {auc:.4f}")
    
    # Save Checkpoint
    output_path = PROJECT_ROOT / "outputs" / "models" / f"streaming_{args.model}_model.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\n  Saved model checkpoint to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
