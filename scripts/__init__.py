# from scripts.main_strong import main, display_results
from model import UNetEfficientNet
from data import SaltDataset
from engine import train_model
from metrics import DiceBCELoss, iou_score, pixel_accuracy, frequency_weighted_iou, WeakFocalLoss, weighted_balanced_accuracy, BinaryWeakFocalLoss
from utils import (compute_coverage, cov_to_class,
                    plot_training_metrics, seed_everything, plot_predictions)
from efficientnet_pytorch import EfficientNet
from data import image_transform, mask_transform       
