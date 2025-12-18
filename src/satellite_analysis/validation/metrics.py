"""
Validation metrics for land cover classification.

Provides accuracy assessment tools comparing classifications
against reference data (e.g., ESA Scene Classification Layer).

Metrics included:
- Overall Accuracy (OA)
- Cohen's Kappa coefficient
- F1-score (per class and weighted)
- Producer's and User's accuracy
- Confusion matrix
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Overall Accuracy (OA).
    
    OA = (number of correctly classified pixels) / (total pixels)
    
    Args:
        y_true: Ground truth labels (flattened or 2D)
        y_pred: Predicted labels (same shape as y_true)
        
    Returns:
        Overall accuracy as float between 0 and 1
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shape mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}")
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    
    accuracy = correct / total if total > 0 else 0.0
    
    logger.info(f"Overall Accuracy: {accuracy:.4f} ({correct:,}/{total:,})")
    
    return float(accuracy)


def compute_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Cohen's Kappa coefficient.
    
    Kappa measures agreement between classification and reference,
    accounting for chance agreement.
    
    Kappa = (OA - Pe) / (1 - Pe)
    where Pe = expected agreement by chance
    
    Interpretation:
    - < 0: Less than chance agreement
    - 0.01–0.20: Slight agreement
    - 0.21–0.40: Fair agreement
    - 0.41–0.60: Moderate agreement
    - 0.61–0.80: Substantial agreement
    - 0.81–1.00: Almost perfect agreement
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Kappa coefficient as float between -1 and 1
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    n = len(y_true)
    if n == 0:
        return 0.0
    
    # Get all unique classes
    classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    
    # Build confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, classes)
    
    # Overall accuracy (observed agreement)
    po = np.trace(cm) / n
    
    # Expected agreement by chance
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)
    
    # Kappa
    if pe == 1.0:
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1 - pe)
    
    logger.info(f"Cohen's Kappa: {kappa:.4f}")
    
    return float(kappa)


def compute_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    average: Optional[str] = 'weighted'
) -> Union[float, Dict[int, float]]:
    """
    Compute F1-score per class or averaged.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional dict mapping class_id to name for logging
        average: Averaging method:
            - None: Return dict with F1 per class
            - 'weighted': Weighted average by class frequency
            - 'macro': Simple average across classes
            - 'micro': Global precision/recall (same as OA for multiclass)
            
    Returns:
        F1 score(s) as float or dict
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    
    # Per-class metrics
    f1_scores = {}
    precision_scores = {}
    recall_scores = {}
    support = {}
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_scores[int(cls)] = f1
        precision_scores[int(cls)] = precision
        recall_scores[int(cls)] = recall
        support[int(cls)] = int(np.sum(y_true == cls))
    
    # Log per-class results
    if class_names:
        logger.info("Per-class F1 scores:")
        for cls in sorted(f1_scores.keys()):
            name = class_names.get(cls, f"Class {cls}")
            logger.info(f"  {name}: F1={f1_scores[cls]:.4f}, "
                       f"P={precision_scores[cls]:.4f}, R={recall_scores[cls]:.4f}, "
                       f"Support={support[cls]:,}")
    
    # Return based on average type
    if average is None:
        return f1_scores
    
    if average == 'micro':
        # Global TP, FP, FN
        tp_total = sum(
            np.sum((y_true == cls) & (y_pred == cls)) for cls in classes
        )
        fp_total = sum(
            np.sum((y_true != cls) & (y_pred == cls)) for cls in classes
        )
        fn_total = sum(
            np.sum((y_true == cls) & (y_pred != cls)) for cls in classes
        )
        
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return float(f1)
    
    if average == 'macro':
        return float(np.mean(list(f1_scores.values())))
    
    if average == 'weighted':
        total_support = sum(support.values())
        if total_support == 0:
            return 0.0
        weighted = sum(f1_scores[cls] * support[cls] for cls in f1_scores) / total_support
        return float(weighted)
    
    raise ValueError(f"Unknown average method: {average}")


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: Optional array of class labels (sorted)
        
    Returns:
        Confusion matrix as 2D numpy array
        Rows = true labels, Columns = predicted labels
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if classes is None:
        classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    
    classes = np.sort(classes)
    n_classes = len(classes)
    
    # Create mapping from class label to index
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Build confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    for t, p in zip(y_true, y_pred):
        if t in class_to_idx and p in class_to_idx:
            cm[class_to_idx[t], class_to_idx[p]] += 1
    
    return cm


def compute_producer_user_accuracy(
    confusion_matrix: np.ndarray,
    class_names: Optional[Dict[int, str]] = None
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute Producer's and User's accuracy from confusion matrix.
    
    Producer's Accuracy (Recall): How well reference pixels are classified
        PA = TP / (TP + FN) = diagonal / column sum
    
    User's Accuracy (Precision): How reliable is the classification
        UA = TP / (TP + FP) = diagonal / row sum
    
    Args:
        confusion_matrix: Confusion matrix (true x predicted)
        class_names: Optional class name mapping
        
    Returns:
        Tuple of (producer_accuracy, user_accuracy) dicts
    """
    cm = np.asarray(confusion_matrix)
    n_classes = cm.shape[0]
    
    producer_acc = {}
    user_acc = {}
    
    for i in range(n_classes):
        col_sum = np.sum(cm[:, i])
        row_sum = np.sum(cm[i, :])
        diagonal = cm[i, i]
        
        producer_acc[i] = diagonal / col_sum if col_sum > 0 else 0.0
        user_acc[i] = diagonal / row_sum if row_sum > 0 else 0.0
    
    if class_names:
        logger.info("Producer's and User's Accuracy:")
        for i in sorted(producer_acc.keys()):
            name = class_names.get(i, f"Class {i}")
            logger.info(f"  {name}: PA={producer_acc[i]:.4f}, UA={user_acc[i]:.4f}")
    
    return producer_acc, user_acc


class ValidationReport:
    """
    Comprehensive validation report for classification results.
    
    Computes all metrics and generates summary statistics.
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize validation report.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: Optional dict mapping class_id to name
        """
        self.y_true = np.asarray(y_true).flatten()
        self.y_pred = np.asarray(y_pred).flatten()
        self.class_names = class_names or {}
        
        # Get classes
        self.classes = np.sort(np.union1d(
            np.unique(self.y_true),
            np.unique(self.y_pred)
        ))
        
        # Compute all metrics
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute all validation metrics."""
        logger.info("Computing validation metrics...")
        
        # Overall accuracy
        self.overall_accuracy = compute_accuracy(self.y_true, self.y_pred)
        
        # Kappa
        self.kappa = compute_kappa(self.y_true, self.y_pred)
        
        # Confusion matrix
        self.confusion_matrix = compute_confusion_matrix(
            self.y_true, self.y_pred, self.classes
        )
        
        # F1 scores
        self.f1_weighted = compute_f1_scores(
            self.y_true, self.y_pred, self.class_names, average='weighted'
        )
        self.f1_macro = compute_f1_scores(
            self.y_true, self.y_pred, self.class_names, average='macro'
        )
        self.f1_per_class = compute_f1_scores(
            self.y_true, self.y_pred, self.class_names, average=None
        )
        
        # Producer/User accuracy
        self.producer_accuracy, self.user_accuracy = compute_producer_user_accuracy(
            self.confusion_matrix, self.class_names
        )
        
        logger.info("Validation metrics computed successfully")
    
    def get_summary(self) -> Dict:
        """
        Get summary dictionary of all metrics.
        
        Returns:
            Dict with all computed metrics
        """
        return {
            'overall_accuracy': self.overall_accuracy,
            'kappa': self.kappa,
            'f1_weighted': self.f1_weighted,
            'f1_macro': self.f1_macro,
            'f1_per_class': self.f1_per_class,
            'producer_accuracy': self.producer_accuracy,
            'user_accuracy': self.user_accuracy,
            'n_samples': len(self.y_true),
            'n_classes': len(self.classes),
            'classes': self.classes.tolist()
        }
    
    def print_report(self) -> str:
        """
        Generate and print formatted validation report.
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CLASSIFICATION VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Summary metrics
        lines.append("📊 SUMMARY METRICS")
        lines.append("-" * 40)
        lines.append(f"Overall Accuracy:  {self.overall_accuracy:.4f} ({self.overall_accuracy*100:.2f}%)")
        lines.append(f"Cohen's Kappa:     {self.kappa:.4f}")
        lines.append(f"F1 (weighted):     {self.f1_weighted:.4f}")
        lines.append(f"F1 (macro):        {self.f1_macro:.4f}")
        lines.append(f"Total Samples:     {len(self.y_true):,}")
        lines.append("")
        
        # Kappa interpretation
        if self.kappa < 0:
            interp = "Less than chance"
        elif self.kappa <= 0.20:
            interp = "Slight agreement"
        elif self.kappa <= 0.40:
            interp = "Fair agreement"
        elif self.kappa <= 0.60:
            interp = "Moderate agreement"
        elif self.kappa <= 0.80:
            interp = "Substantial agreement"
        else:
            interp = "Almost perfect agreement"
        lines.append(f"Kappa Interpretation: {interp}")
        lines.append("")
        
        # Per-class metrics
        lines.append("📈 PER-CLASS METRICS")
        lines.append("-" * 40)
        lines.append(f"{'Class':<20} {'F1':>8} {'PA':>8} {'UA':>8} {'Support':>10}")
        lines.append("-" * 40)
        
        for cls in self.classes:
            name = self.class_names.get(cls, f"Class {cls}")
            f1 = self.f1_per_class.get(int(cls), 0.0)
            pa = self.producer_accuracy.get(int(cls), 0.0)
            ua = self.user_accuracy.get(int(cls), 0.0)
            support = np.sum(self.y_true == cls)
            
            lines.append(f"{name:<20} {f1:>8.4f} {pa:>8.4f} {ua:>8.4f} {support:>10,}")
        
        lines.append("")
        lines.append("PA = Producer's Accuracy (Recall)")
        lines.append("UA = User's Accuracy (Precision)")
        lines.append("")
        
        # Confusion matrix
        lines.append("🔢 CONFUSION MATRIX")
        lines.append("-" * 40)
        lines.append("Rows = True labels, Columns = Predicted")
        lines.append("")
        
        # Header
        header = "         " + " ".join(f"{cls:>7}" for cls in self.classes)
        lines.append(header)
        
        # Rows
        for i, cls in enumerate(self.classes):
            name = self.class_names.get(cls, f"C{cls}")[:7]
            row = f"{name:<8} " + " ".join(f"{self.confusion_matrix[i, j]:>7,}" for j in range(len(self.classes)))
            lines.append(row)
        
        lines.append("")
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        print(report)
        
        return report
