# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository implements Domain-Adversarial Neural Networks (DANN) for domain adaptation on the PACS dataset. It contains a research project that applies domain adaptation techniques to transfer learning between different visual domains (photo, art painting, cartoon, sketch).

## Key Components

### Main Implementation
- `code/main.py`: Primary training script that implements the complete DANN pipeline
- `code/models/models.py`: Contains the DANN_AlexNet architecture with gradient reversal layer
- `code/utils/utils.py`: Utility functions for visualization and data preprocessing

### Dataset Structure
- `Homework3-PACS/PACS/`: Contains the PACS dataset organized by domain (photo, art_painting, cartoon, sketch) and class (dog, elephant, giraffe, guitar, horse, house, person)

### Documentation
- `README.md`: Main project documentation with architecture overview and requirements
- `report/`: Contains detailed analysis and results
- `assignment/`: Assignment instructions and requirements

## Running the Code

### New Command-Line Interface
The repository now includes a modular training script with command-line interface:

#### Quick Start
```bash
# Navigate to code directory
cd code/

# Quick test (2 epochs, small batch)
python train.py --quick-test

# Run with predefined configs
python train.py --config configs/baseline_no_dann.json
python train.py --config configs/photo_to_art_dann.json

# Interactive menu (Windows/Linux)
./run_experiments.sh    # Linux/Mac
run_experiments.bat     # Windows
```

#### Command Line Arguments
```bash
python train.py [options]

# Core options:
--mode {3A,3B,4A,4C}     # Training mode
--source-domain          # photo, art, cartoon, sketch  
--target-domain          # photo, art, cartoon, sketch
--batch-size INT         # Batch size (default: 128)
--lr FLOAT              # Learning rate (default: 0.01)  
--epochs INT            # Training epochs (default: 30)
--alpha FLOAT           # Domain adaptation weight (default: 0.25)

# Convenience options:
--quick-test            # Fast testing (reduced epochs/batch)
--config PATH           # Load JSON config file
--output-dir PATH       # Save results directory
--save-model           # Save trained model weights
--show-plots           # Display training plots
```

#### Available Configurations
- `configs/baseline_no_dann.json`: Standard training without domain adaptation
- `configs/photo_to_art_dann.json`: DANN from Photo to Art painting
- `configs/photo_to_sketch_dann.json`: DANN from Photo to Sketch  
- `configs/quick_test.json`: Fast testing configuration

### Original Notebook/Script Requirements
The project requires Python 3.7.12 with the following packages:
```
torch==1.3.1
torchvision==0.5.0
numpy
matplotlib
PIL
tqdm
```

### Training Modes
- **MODE 3A**: Training without domain adaptation (baseline)
- **MODE 3B**: DANN training Photo→Art painting with domain adaptation
- **MODE 4A**: Grid search Photo→Cartoon/Sketch without domain adaptation  
- **MODE 4C**: Grid search Photo→Cartoon/Sketch with domain adaptation

### Key Hyperparameters
- `BATCH_SIZE`: Batch size for training (typically 128 or 256)
- `LR`: Learning rate (typically 0.01 or 0.0001)  
- `ALPHA`: Domain adaptation weight (typically 0.25 or 0.5)
- `NUM_EPOCHS`: Training epochs (typically 30)

## Architecture Details

### DANN Implementation
- Uses modified AlexNet as feature extractor
- Implements gradient reversal layer (`ReverseLayerF`) for domain-adversarial training
- Contains separate classifier (7 classes) and domain discriminator (2 domains)
- Pretrained weights from ImageNet are used and adapted

### Data Pipeline
- Images preprocessed with ImageNet normalization: means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)
- Center crop to 224x224 for AlexNet input requirements
- No data augmentation applied in current implementation

## Development Notes

- The codebase is primarily structured for research/experimental use
- Original code designed for Google Colab environment (includes git cloning of datasets)
- No formal test suite - evaluation done through accuracy metrics
- Visualization functions available for loss curves and data distribution analysis