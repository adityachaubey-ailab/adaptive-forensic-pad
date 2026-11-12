import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import yaml
import torch
import logging
from torchvision import transforms

from src.models.pad_model import PADBaselineModel
from src.data.dataset import get_dataloaders
from src.training.trainer import PADTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train PAD Engine")
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--metadata-csv', type=str, default='dataset/metadata.csv', help='Path to metadata CSV')
    parser.add_argument('--model-type', type=str, default='small', choices=['small', 'large'], help='MobileNetV3 size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/models', help='Checkpoint output directory')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = {
        'metadata_csv': args.metadata_csv,
        'batch_size': args.batch_size,
        'num_workers': 0, # Windows multiprocessing safety
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'checkpoint_dir': args.checkpoint_dir,
        'use_amp': torch.cuda.is_available(),
        'dataset_type': 'image'
    }
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            user_cfg = yaml.safe_load(f)
            if user_cfg:
                config.update(user_cfg)
                
    device = torch.device(args.device)
    logger.info(f"Initializing PAD training on {device}...")
    
    # Static torchvision transforms
    train_transform = None
    val_transform = None
    
    # Initialize Dataloaders
    loaders = get_dataloaders(config)
    
    if 'train' not in loaders or len(loaders['train']) == 0:
        logger.error("No training data found in metadata.csv!")
        return
        
    train_loader = loaders['train']
    val_loader = loaders.get('validation', train_loader)
    
    # Build Model
    model = PADBaselineModel(model_type=args.model_type, pretrained=True, num_classes=2)
    
    # Initialize Trainer
    trainer = PADTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    logger.info(f"Starting training for {config['epochs']} epochs...")
    history = trainer.train(config['epochs'])
    
    # Save final model checkpoint
    out_dir = Path(config['checkpoint_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    final_pth = out_dir / "streaming_baseline_model.pth"
    torch.save(model.state_dict(), final_pth)
    logger.info(f"Saved trained model checkpoint to {final_pth}")
    
    print("\n" + "=" * 50)
    print("  PAD ENGINE TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Best ACER:    {trainer.best_acer:.4f}")
    print(f"  Checkpoint:   {final_pth}")
    print("=" * 50)

if __name__ == '__main__':
    main()
