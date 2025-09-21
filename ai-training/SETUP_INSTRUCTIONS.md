# 🚀 Setup Instructions - Signature Verification AI System

## 📋 **System Requirements**

### **Local PC (Windows)**
- **Python**: 3.10.11 ✅ (Keep this version)
- **Purpose**: Development and CPU training
- **CUDA**: Not needed (CPU-only)

### **GPU Instance (AWS)**
- **Python**: 3.12.6 ✅ (Pre-installed on Deep Learning AMI)
- **Purpose**: GPU training acceleration
- **CUDA**: Pre-installed and configured

## 🖥️ **Local PC Setup (Windows)**

### **1. Install Dependencies**
```bash
# Navigate to ai-training folder
cd ai-training

# Install CPU-only dependencies
pip install -r requirements-local.txt
```

### **2. Verify Installation**
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Expected Output:**
```
TensorFlow version: 2.18.0
GPUs: []
```

### **3. Start Development Server**
```bash
# Start FastAPI backend
python main.py

# In another terminal, start React frontend
npm run dev
```

## ☁️ **GPU Instance Setup (AWS)**

### **1. Connect to Instance**
```bash
ssh -i "your-key.pem" ubuntu@your-instance-ip
```

### **2. Activate TensorFlow Environment**
```bash
source /opt/tensorflow/bin/activate
```

### **3. Clone Repository**
```bash
git clone https://github.com/jesriel914-ai/hello-world-magic-94.git
cd hello-world-magic-94
git pull origin cursor/greeting-or-placeholder-task-0485
```

### **4. Install GPU Dependencies**
```bash
cd ai-training
pip install -r requirements-gpu.txt
```

### **5. Verify GPU Setup**
```bash
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Expected Output:**
```
TensorFlow version: 2.18.1
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 🎯 **Usage Instructions**

### **Local Development (CPU Training)**
1. **Upload signatures** via web interface
2. **Train models** using CPU (slower but works)
3. **Test verification** with trained models
4. **Deploy to production** when ready

### **Production Training (GPU)**
1. **Upload signatures** via web interface
2. **System automatically** launches GPU instance
3. **GPU training** runs automatically (10-50x faster)
4. **Results** saved to S3 and database
5. **Instance terminates** automatically

## 🔧 **Troubleshooting**

### **Local PC Issues**
- **CUDA errors**: Use `requirements-local.txt` (CPU-only)
- **Python version**: Keep 3.10.11 (compatible with all dependencies)
- **Memory issues**: Reduce batch size in training

### **GPU Instance Issues**
- **TensorFlow not found**: Activate environment with `source /opt/tensorflow/bin/activate`
- **GPU not detected**: Check `nvidia-smi` output
- **Permission errors**: Use `sudo` for system-level operations

## 📊 **Performance Comparison**

| Training Type | Speed | Use Case |
|---------------|-------|----------|
| **CPU (Local)** | 30-60 minutes | Development, testing |
| **GPU (AWS)** | 2-5 minutes | Production, large datasets |

## ✅ **Ready to Use!**

Your system is now properly configured for:
- ✅ **Local development** with CPU training
- ✅ **Production deployment** with GPU acceleration
- ✅ **Real school attendance** tracking
- ✅ **Capstone project** presentation

**No more CUDA errors on your Windows PC!** 🎉