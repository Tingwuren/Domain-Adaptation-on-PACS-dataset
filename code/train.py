#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DANN Training Script with Command Line Interface
Domain Adaptation on PACS Dataset
"""

import argparse
import os
import sys
import logging
import json
import numpy as np
import random
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends import cudnn

import torchvision
from torchvision import transforms
from tqdm import tqdm

from models.models import dann_net
from utils.utils import plotImageDistribution, plotLosses

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    
    # Make cudnn deterministic (may impact performance)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logging(log_level='INFO', output_dir='./outputs'):
    """Setup logging configuration"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file path (simple name)
    log_path = os.path.join(output_dir, 'training.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_path}")
    return logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DANN Training on PACS Dataset')
    
    # Training strategy
    parser.add_argument('--dann', action='store_true', default=False,
                       help='Use Domain-Adversarial Neural Networks (DANN) for domain adaptation')
    parser.add_argument('--no-dann', dest='dann', action='store_false',
                       help='Do not use domain adaptation (baseline training only)')
    
    # Dataset parameters
    parser.add_argument('--data-root', type=str, default='./Homework3-PACS',
                       help='Path to PACS dataset root')
    parser.add_argument('--source-domain', type=str, default='photo',
                       choices=['photo', 'art', 'cartoon', 'sketch'],
                       help='Source domain for training')
    parser.add_argument('--target-domain', type=str, default='sketch',
                       choices=['photo', 'art', 'cartoon', 'sketch'],
                       help='Target domain for adaptation/testing')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--alpha', type=float, default=0.25,
                       help='Domain adaptation weight (default: 0.25)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                       help='Weight decay (default: 5e-5)')
    parser.add_argument('--step-size', type=int, default=20,
                       help='Learning rate step size (default: 20)')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Learning rate decay gamma (default: 0.1)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ImageNet weights (default: True)')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Do not use pretrained weights, train from scratch')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Device and performance
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training (default: auto)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    
    # Output and logging
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Directory to save outputs (default: ./outputs)')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--log-frequency', type=int, default=1,
                       help='Log frequency in epochs (default: 1)')
    
    # Visualization and evaluation
    parser.add_argument('--show-plots', action='store_true',
                       help='Show visualization plots')
    parser.add_argument('--eval-train', action='store_true',
                       help='Evaluate accuracy on training set')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (reduced epochs and batch size)')
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help='Path to JSON configuration file')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(args, output_dir):
    """Save configuration to output directory"""
    config_dict = vars(args)
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def setup_device(device_arg):
    """Setup computing device"""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    return device

def get_domain_datasets(data_root, transform):
    """Load domain datasets"""
    domain_paths = {
        'photo': os.path.join(data_root, 'PACS', 'photo'),
        'art': os.path.join(data_root, 'PACS', 'art_painting'),
        'cartoon': os.path.join(data_root, 'PACS', 'cartoon'),
        'sketch': os.path.join(data_root, 'PACS', 'sketch')
    }
    
    datasets = {}
    for domain, path in domain_paths.items():
        if os.path.exists(path):
            datasets[domain] = torchvision.datasets.ImageFolder(path, transform=transform)
        else:
            print(f"Warning: Domain path {path} not found")
    
    return datasets

def create_data_loaders(datasets, source_domain, target_domain, batch_size, num_workers):
    """Create data loaders for training"""
    source_loader = DataLoader(
        datasets[source_domain], 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True
    )
    
    target_loader = DataLoader(
        datasets[target_domain], 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=False
    )
    
    return source_loader, target_loader

def train_epoch(net, source_loader, target_loader, optimizer, criterion, device, args, logger):
    """Train for one epoch"""
    net.train()
    
    epoch_class_loss = []
    epoch_source_loss = []
    epoch_target_loss = []
    
    # Get target data iterator for domain adaptation
    if args.dann:  # Domain adaptation mode
        target_iter = iter(target_loader)
    
    for batch_idx, (source_images, source_labels) in enumerate(tqdm(source_loader, desc="Training")):
        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        
        optimizer.zero_grad()
        
        # Step 1: Train classifier on source data
        class_outputs = net(source_images)
        loss_class = criterion(class_outputs, source_labels)
        epoch_class_loss.append(loss_class.item())
        
        loss_class.backward()
        
        total_loss = loss_class
        
        # Step 2 & 3: Domain adaptation
        if args.dann:
            try:
                target_images, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_images, _ = next(target_iter)
            
            target_images = target_images.to(device)
            
            # Ensure batch sizes match for domain discriminator
            min_batch_size = min(source_images.size(0), target_images.size(0))
            source_domain_batch = source_images[:min_batch_size]
            target_domain_batch = target_images[:min_batch_size]
            
            # Train domain discriminator with source data (label = 0)
            source_domain_outputs = net.forward(source_domain_batch, alpha=args.alpha)
            labels_discr_source = torch.zeros(min_batch_size, dtype=torch.int64).to(device)
            loss_discr_source = criterion(source_domain_outputs, labels_discr_source)
            epoch_source_loss.append(loss_discr_source.item())
            loss_discr_source.backward()
            
            # Train domain discriminator with target data (label = 1)
            target_domain_outputs = net.forward(target_domain_batch, alpha=args.alpha)
            labels_discr_target = torch.ones(min_batch_size, dtype=torch.int64).to(device)
            loss_discr_target = criterion(target_domain_outputs, labels_discr_target)
            epoch_target_loss.append(loss_discr_target.item())
            loss_discr_target.backward()
            
            # Add domain adaptation losses with proper weighting
            # domain_loss = (loss_discr_source + loss_discr_target) * args.alpha
            # total_loss = total_loss + domain_loss
        
        # Backward pass and optimization
        # total_loss.backward()
        optimizer.step()
    
    return epoch_class_loss, epoch_source_loss, epoch_target_loss

def evaluate(net, data_loader, device, dataset_name, criterion=None):
    """Evaluate model on a dataset"""
    net.eval()
    running_corrects = 0
    total_samples = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluating on {dataset_name}"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = net(images)
            _, preds = torch.max(outputs.data, 1)
            
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += labels.size(0)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
    
    accuracy = running_corrects / total_samples
    avg_loss = running_loss / total_samples if criterion is not None else None
    
    if avg_loss is not None:
        return accuracy, avg_loss
    else:
        return accuracy

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Load config file if provided
    config_file_name = None
    if args.config:
        config = load_config(args.config)
        # Extract config filename without extension
        config_file_name = os.path.splitext(os.path.basename(args.config))[0]
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)
    
    # Quick test mode adjustments
    if args.quick_test:
        args.epochs = min(5, args.epochs)
        args.batch_size = min(32, args.batch_size)
        print("Quick test mode: Reduced epochs and batch size")
    
    # Create timestamped output directory
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if config_file_name:
        # Use config file name as base directory
        base_output_dir = f"./outputs/{config_file_name}"
        args.output_dir = os.path.join(base_output_dir, timestamp)
    else:
        # Use default output directory with timestamp
        args.output_dir = f"./outputs/training_{timestamp}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Setup logging (after output directory is created)
    logger = setup_logging(args.log_level, args.output_dir)
    logger.info(f"Starting training with DANN: {'Enabled' if args.dann else 'Disabled'}")
    logger.info(f"Random seed set to: {args.seed}")
    logger.info(f"Output directory: {args.output_dir}")
    
    save_config(args, args.output_dir)
    
    # Setup device
    device = setup_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Data preprocessing
    means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
    
    # Load datasets
    logger.info("Loading datasets...")
    datasets = get_domain_datasets(args.data_root, transform)
    
    if not datasets:
        logger.error("No datasets found. Please check data path.")
        return
    
    # Create data loaders
    source_loader, target_loader = create_data_loaders(
        datasets, args.source_domain, args.target_domain, 
        args.batch_size, args.num_workers
    )
    
    # Create model
    logger.info("Creating DANN model...")
    net = dann_net(pretrained=args.pretrained).to(device)
    if args.pretrained:
        logger.info("Using pretrained ImageNet weights")
    else:
        logger.info("Training from scratch (no pretrained weights)")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, 
                         momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # Training loop
    logger.info("Starting training...")
    all_class_losses = []
    all_source_losses = []
    all_target_losses = []
    
    # CSV logging setup
    csv_path = os.path.join(args.output_dir, 'training_metrics.csv')
    csv_fieldnames = ['epoch', 'lr', 'avg_class_loss', 'avg_source_loss', 'avg_target_loss', 'val_accuracy', 'val_loss']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        csv_writer.writeheader()
        
        for epoch in range(args.epochs):
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch+1}/{args.epochs}, LR = {current_lr:.6f}")
            
            # Train for one epoch
            class_losses, source_losses, target_losses = train_epoch(
                net, source_loader, target_loader, optimizer, criterion, device, args, logger
            )
            
            all_class_losses.extend(class_losses)
            all_source_losses.extend(source_losses)
            all_target_losses.extend(target_losses)
            
            # Calculate average losses for this epoch
            avg_class_loss = np.mean(class_losses)
            avg_source_loss = np.mean(source_losses) if source_losses else 0.0
            avg_target_loss = np.mean(target_losses) if target_losses else 0.0
            
            # Log average losses for this epoch
            logger.info(f"Epoch {epoch+1} - Avg Classifier Loss: {avg_class_loss:.4f}")
            
            if args.dann and source_losses and target_losses:
                logger.info(f"Epoch {epoch+1} - Avg Source Domain Loss: {avg_source_loss:.4f}")
                logger.info(f"Epoch {epoch+1} - Avg Target Domain Loss: {avg_target_loss:.4f}")
            
            # Validation - always evaluate every epoch
            val_result = evaluate(net, target_loader, device, args.target_domain, criterion)
            if isinstance(val_result, tuple):
                val_acc, val_loss = val_result
            else:
                val_acc = val_result
                val_loss = None
            
            logger.info(f"Validation accuracy on {args.target_domain}: {val_acc:.4f}")
            if val_loss is not None:
                logger.info(f"Validation loss on {args.target_domain}: {val_loss:.4f}")
            
            if args.eval_train:
                train_acc = evaluate(net, source_loader, device, args.source_domain)
                logger.info(f"Training accuracy: {train_acc:.4f}")
            
            # Write to CSV
            csv_row = {
                'epoch': epoch + 1,
                'lr': current_lr,
                'avg_class_loss': avg_class_loss,
                'avg_source_loss': avg_source_loss,
                'avg_target_loss': avg_target_loss,
                'val_accuracy': val_acc,
                'val_loss': val_loss if val_loss is not None else 0.0
            }
            csv_writer.writerow(csv_row)
            csvfile.flush()  # Ensure data is written immediately
            
            scheduler.step()
    
    logger.info(f"Training metrics saved to: {csv_path}")
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_result = evaluate(net, target_loader, device, args.target_domain, criterion)
    if isinstance(final_result, tuple):
        final_accuracy, final_loss = final_result
        logger.info(f"Final accuracy on {args.target_domain}: {final_accuracy:.4f}")
        logger.info(f"Final loss on {args.target_domain}: {final_loss:.4f}")
    else:
        final_accuracy = final_result
        logger.info(f"Final accuracy on {args.target_domain}: {final_accuracy:.4f}")
    
    # Save model
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'dann_model.pth')
        torch.save(net.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    # Plot losses
    if args.show_plots and all_source_losses and all_target_losses:
        loss_plot_path = plotLosses(all_class_losses, all_source_losses, all_target_losses, 
                  n_epochs=len(all_class_losses), output_dir=args.output_dir, show=True)
        logger.info(f"Loss plot saved to: {loss_plot_path}")
    elif all_class_losses:  # Always save loss plot even if not showing
        loss_plot_path = plotLosses(all_class_losses, all_source_losses or [], all_target_losses or [], 
                  n_epochs=len(all_class_losses), output_dir=args.output_dir, show=False)
        logger.info(f"Loss plot saved to: {loss_plot_path}")
    
    # Log final loss statistics
    if all_class_losses:
        logger.info(f"Final Training Statistics:")
        logger.info(f"  Final Classifier Loss: {all_class_losses[-1]:.4f}")
        logger.info(f"  Average Classifier Loss: {np.mean(all_class_losses):.4f}")
        
        if all_source_losses and all_target_losses:
            logger.info(f"  Final Source Domain Loss: {all_source_losses[-1]:.4f}")
            logger.info(f"  Final Target Domain Loss: {all_target_losses[-1]:.4f}")
            logger.info(f"  Average Source Domain Loss: {np.mean(all_source_losses):.4f}")
            logger.info(f"  Average Target Domain Loss: {np.mean(all_target_losses):.4f}")
        else:
            logger.info(f"  Domain adaptation not used (--dann not specified)")
    
    # Save results
    results = {
        'final_accuracy': final_accuracy,
        'config': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()