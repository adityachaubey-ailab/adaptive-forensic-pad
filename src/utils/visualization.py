import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from typing import List, Dict, Optional, Any
from sklearn.metrics import roc_curve, auc, confusion_matrix, det_curve

# Set professional style
sns.set_theme(style="whitegrid")

def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: Optional[str] = None):
    """Plots ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_det_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: Optional[str] = None):
    """Plots DET curve."""
    fpr, fnr, _ = det_curve(y_true, y_scores)
    
    plt.figure()
    plt.plot(fpr, fnr, lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('False Negative Rate (log scale)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
    """Plots confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plots loss/metric curves over epochs."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if 'train_loss' in history:
        plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if 'train_acc' in history:
        plt.plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history:
        plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_score_distribution(scores_real: np.ndarray, scores_spoof: np.ndarray, save_path: Optional[str] = None):
    """Plots score histogram for real vs spoof."""
    plt.figure(figsize=(8, 6))
    sns.histplot(scores_real, color="green", label="Real", kde=True, stat="density", bins=30, alpha=0.5)
    sns.histplot(scores_spoof, color="red", label="Spoof", kde=True, stat="density", bins=30, alpha=0.5)
    
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_apcer_bpcer_tradeoff(y_true: np.ndarray, y_scores: np.ndarray, save_path: Optional[str] = None):
    """Plots APCER vs BPCER at different thresholds."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    apcer = fpr  # Attack Presentation Classification Error Rate (Spoofs classified as Bona Fide)
    bpcer = 1 - tpr  # Bona Fide Presentation Classification Error Rate (Bona Fide classified as Spoof)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, apcer, label='APCER (Spoof as Real)', color='red')
    plt.plot(thresholds, bpcer, label='BPCER (Real as Spoof)', color='green')
    
    # HTER
    hter = (apcer + bpcer) / 2
    plt.plot(thresholds, hter, label='HTER', color='blue', linestyle='--')
    
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('APCER vs BPCER Tradeoff')
    plt.xlim([0, 1])
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_evidence(frame: np.ndarray, evidence_dict: Dict[str, float], save_path: Optional[str] = None) -> np.ndarray:
    """Overlays head scores on frame."""
    out_frame = frame.copy()
    h, w = out_frame.shape[:2]
    
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    for key, score in evidence_dict.items():
        if key == 'final_decision':
            continue
            
        # Determine color based on score
        if score > 0.7:
            color = (0, 255, 0) # Green for real
        elif score < 0.3:
            color = (0, 0, 255) # Red for spoof
        else:
            color = (0, 255, 255) # Yellow for uncertain
            
        text = f"{key}: {score:.2f}"
        cv2.putText(out_frame, text, (10, y_offset), font, font_scale, color, thickness)
        
        # Draw bar
        bar_x = 250
        bar_w = 100
        bar_h = 15
        cv2.rectangle(out_frame, (bar_x, y_offset - 15), (bar_x + bar_w, y_offset), (200, 200, 200), -1)
        cv2.rectangle(out_frame, (bar_x, y_offset - 15), (bar_x + int(bar_w * score), y_offset), color, -1)
        
        y_offset += 30
        
    if 'final_decision' in evidence_dict:
        score = evidence_dict.get('confidence', 0.0)
        decision = evidence_dict['final_decision']
        color = (0, 255, 0) if decision == 'Real' else (0, 0, 255)
        text = f"DECISION: {decision} ({score:.2f})"
        cv2.putText(out_frame, text, (10, h - 30), font, 1.0, color, 3)
        
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))
        
    return out_frame

def create_benchmark_dashboard(benchmark_results: Dict[str, Any], save_path: Optional[str] = None):
    """Creates a multi-panel figure for benchmark results."""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Benchmark Dashboard', fontsize=16)
    
    # 1. Latency breakdown bar chart
    ax1 = plt.subplot(2, 2, 1)
    if 'latency' in benchmark_results:
        latency_data = benchmark_results['latency']
        names = list(latency_data.keys())
        values = list(latency_data.values())
        sns.barplot(x=names, y=values, ax=ax1)
        ax1.set_title('Latency Breakdown (ms)')
        ax1.set_ylabel('Time (ms)')
        
    # 2. Score distributions
    ax2 = plt.subplot(2, 2, 2)
    if 'scores_real' in benchmark_results and 'scores_spoof' in benchmark_results:
        sns.histplot(benchmark_results['scores_real'], color="green", label="Real", kde=True, ax=ax2, alpha=0.5)
        sns.histplot(benchmark_results['scores_spoof'], color="red", label="Spoof", kde=True, ax=ax2, alpha=0.5)
        ax2.set_title('Score Distribution')
        ax2.legend()
        
    # 3. ROC curve
    ax3 = plt.subplot(2, 2, 3)
    if 'y_true' in benchmark_results and 'y_scores' in benchmark_results:
        fpr, tpr, _ = roc_curve(benchmark_results['y_true'], benchmark_results['y_scores'])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax3.set_title('ROC Curve')
        ax3.legend(loc="lower right")
        
    # 4. Accuracy metrics table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    if 'metrics' in benchmark_results:
        metrics = benchmark_results['metrics']
        cell_text = [[f"{k}", f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in metrics.items()]
        table = ax4.table(cellText=cell_text, colLabels=['Metric', 'Value'], loc='center', cellLoc='left')
        table.scale(1, 2)
        table.set_fontsize(14)
        ax4.set_title('Accuracy Metrics')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
