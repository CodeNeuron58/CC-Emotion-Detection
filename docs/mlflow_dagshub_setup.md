# MLflow and DagsHub Integration Setup

## Overview
This document explains how to set up MLflow tracking with DagsHub integration for the emotion detection project.

## Prerequisites
1. DagsHub account (sign up at https://dagshub.com)
2. Create a new repository on DagsHub or connect your existing GitHub repo

## Setup Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure DagsHub Repository
1. Update `params.yaml` with your DagsHub details:
   ```yaml
   tracking:
     mlflow_tracking_uri: "https://dagshub.com/YOUR_USERNAME/CC-Emotion-Detection.mlflow"
     experiment_name: "emotion_detection_experiment"
     dagshub_repo_owner: "YOUR_USERNAME"
     dagshub_repo_name: "CC-Emotion-Detection"
   ```

### 3. Set Environment Variables
Set your DagsHub credentials as environment variables:

**Windows PowerShell:**
```powershell
$env:DAGSHUB_USERNAME = "your_username"
$env:DAGSHUB_TOKEN = "your_dagshub_token"
```

**Linux/Mac:**
```bash
export DAGSHUB_USERNAME="your_username"
export DAGSHUB_TOKEN="your_dagshub_token"
```

### 4. Get DagsHub Token
1. Go to https://dagshub.com/user/settings/tokens
2. Create a new token with appropriate permissions
3. Copy the token and use it as `DAGSHUB_TOKEN`

## Usage

### Option 1: Run Individual Stages with MLflow
```bash
# Model building with MLflow tracking
python src/model/model_building.py

# Model evaluation with MLflow tracking  
python src/model/model_evaluation.py
```

### Option 2: Run Complete Pipeline with MLflow
```bash
# Run the complete pipeline with MLflow tracking
python src/mlflow_pipeline.py
```

### Option 3: Use DVC Pipeline (Traditional)
```bash
# Run DVC pipeline (will include MLflow tracking in individual stages)
dvc repro
```

## What Gets Tracked

### Parameters
- All hyperparameters from `params.yaml`
- Dataset information (number of samples, features)
- Model configuration

### Metrics
- Training accuracy
- Test accuracy, precision, recall, ROC-AUC
- All evaluation metrics

### Artifacts
- Trained model files
- Dataset files
- Evaluation results (metrics.json)
- Pipeline logs

### Models
- Trained models are registered in MLflow Model Registry
- Model versioning and stage management

## Viewing Results

1. **DagsHub Web Interface:**
   - Visit: `https://dagshub.com/YOUR_USERNAME/CC-Emotion-Detection`
   - Navigate to "Experiments" tab to view MLflow runs

2. **Local MLflow UI:**
   ```bash
   mlflow ui
   ```

## Integration Points

### 1. Model Building (`src/model/model_building.py`)
- ✅ MLflow run tracking
- ✅ Parameter logging
- ✅ Model registration
- ✅ Training metrics

### 2. Model Evaluation (`src/model/model_evaluation.py`)
- ✅ Evaluation metrics logging
- ✅ Test dataset information
- ✅ Results artifacts

### 3. Pipeline Management (`src/mlflow_pipeline.py`)
- ✅ End-to-end pipeline tracking
- ✅ DVC integration
- ✅ Error handling and logging

### 4. Utilities (`src/utils/mlflow_utils.py`)
- ✅ MLflow setup and configuration
- ✅ DagsHub initialization
- ✅ Parameter and artifact logging helpers

## Troubleshooting

1. **Authentication Issues:**
   - Verify DAGSHUB_USERNAME and DAGSHUB_TOKEN are set correctly
   - Check token permissions on DagsHub

2. **Connection Issues:**
   - Verify internet connection
   - Check DagsHub repository URL in params.yaml

3. **Import Errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate your virtual environment

## Benefits

1. **Experiment Tracking:** All model experiments are automatically tracked
2. **Reproducibility:** Complete parameter and artifact logging
3. **Model Versioning:** Automatic model registry management
4. **Collaboration:** Team can view experiments on DagsHub
5. **Data Science Workflow:** Integrated with DVC pipeline
6. **Visual Interface:** Rich web UI for experiment comparison