# 🎯 Core Files Summary - AI Training System

## 📁 **Essential Files Only (Keep These)**

### **Main Application**
- `main.py` - FastAPI entry point
- `config.py` - Configuration settings

### **API Layer** 
- `api/training.py` - Training endpoints (CPU + GPU)
- `api/verification.py` - Signature verification
- `api/progress.py` - Real-time progress tracking
- `api/versioning.py` - Model version management
- `api/utils.py` - Utility endpoints
- `api/uploads.py` - File upload handling

### **Core Models**
- `models/signature_embedding_model.py` - Individual student models
- `models/global_signature_model.py` - Multi-student verification
- `models/database.py` - Database operations

### **Essential Utils**
- `utils/aws_gpu_training.py` - **GPU training manager**
- `utils/signature_preprocessing.py` - Image preprocessing
- `utils/image_processing.py` - Image validation
- `utils/s3_storage.py` - S3 operations
- `utils/s3_supabase_sync.py` - Database sync
- `utils/job_queue.py` - Job management
- `utils/training_callback.py` - Training callbacks
- `utils/cpu_optimization.py` - CPU optimization
- `utils/artifacts.py` - Model artifacts
- `utils/tfdata.py` - TensorFlow data utilities

### **Training Scripts**
- `scripts/train_gpu_tf213.py` - **ACTIVE GPU training script**

### **Dependencies**
- `requirements.txt` - Main dependencies
- `requirements-gpu.txt` - GPU-specific dependencies

### **Documentation**
- `GPU_TRAINING_SETUP.md` - Complete setup guide

---

## ❌ **Files to Delete (Duplicates/Unused)**

### **Duplicate Training Scripts**
- `scripts/train_gpu_template.py` - Old template
- `train_gpu.py` - Duplicate

### **Test Files (15+ files)**
- All `test_*.py` files
- All `run_*.py` files
- `comprehensive_test_runner.py`
- `validate_ai_system.py`

### **Documentation (8+ files)**
- `AI_SYSTEM_README.md`
- `AWS_GPU_SETUP*.md` (3 files)
- `EC2_ONLY_README.md`
- `GPU_*.md` (4 files) - Keep only `GPU_TRAINING_SETUP.md`

### **Unused Utils (9 files)**
- `utils/antispoofing.py`
- `utils/augmentation.py`
- `utils/direct_s3_saving.py`
- `utils/enhanced_logging.py`
- `utils/enhanced_s3_upload.py`
- `utils/local_model_saving.py`
- `utils/migration.py`
- `utils/optimized_s3_saving.py`
- `utils/storage.py`

### **Unused Directories**
- `services/` - Not imported
- `sql/` - Not used
- `ai-training/` - Duplicate subfolder
- `tests/` - Too many test files

### **Generated Files**
- `*.keras` files - Can be regenerated
- `training_log.csv` - Generated log
- `run.txt` - Generated file

---

## 🚀 **Core Workflow**

### **1. Image Processing Pipeline**
```
SignaturePreprocessor (utils/signature_preprocessing.py)
    ↓
Image validation (utils/image_processing.py)
    ↓
Augmentation (built into SignaturePreprocessor)
```

### **2. Training Pipeline**
```
API endpoint (api/training.py)
    ↓
GPU Manager (utils/aws_gpu_training.py)
    ↓
Training Script (scripts/train_gpu_tf213.py)
    ↓
Model Saving (utils/s3_storage.py)
```

### **3. Verification Pipeline**
```
API endpoint (api/verification.py)
    ↓
Model Loading (utils/s3_storage.py)
    ↓
Verification (models/global_signature_model.py)
```

---

## 🎯 **Answer to Your Question**

**Which training script is correct?**
- ✅ **`scripts/train_gpu_tf213.py`** - This is the ACTIVE script used by the GPU manager
- ❌ **`scripts/train_gpu_template.py`** - Old template, DELETE
- ❌ **`train_gpu.py`** - Duplicate, DELETE

**What files are not used?**
- 15+ test files
- 8+ documentation files  
- 9+ unused utility files
- 3+ duplicate directories
- Generated model files

**Total files to delete: ~40+ files**