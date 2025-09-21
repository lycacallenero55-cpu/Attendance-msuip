# 🚀 GPU Training Setup Guide

## 🎯 Quick Summary

**For YOU (One-time setup):**
1. **Launch GPU instance** with Deep Learning AMI
2. **Install dependencies** (if using Ubuntu)
3. **Clone repository** and configure environment
4. **Test setup** with provided test script

**For USERS (Automatic):**
1. **Click "Train Model"** button in web interface
2. **Watch real-time progress** - that's it!

**The system automatically handles everything else!**

---

## Overview

Your web application now has **GPU training support**! When users click "Train Model" in the web interface, it will:

1. **Launch AWS GPU instance** (10-50x faster than CPU)
2. **Upload training data** to S3
3. **Train AI models** on GPU with TensorFlow 2.13
4. **Stream real-time progress** to the web interface
5. **Save models** to S3 and database
6. **Clean up** the GPU instance

## ✅ What's Fixed

- ✅ **Updated AMI ID**: Now uses `ami-00f0918871e41f60d` (Deep Learning OSS Nvidia Driver AMI GPU TensorFlow 2.18 Ubuntu 22.04)
- ✅ **TensorFlow 2.13 Compatible**: Updated all APIs for compatibility
- ✅ **GPU Detection**: Fixed `tf.config.list_physical_devices('GPU')`
- ✅ **Training Script**: New `train_gpu_tf213.py` with proper error handling
- ✅ **Real-time Progress**: Web interface shows training progress
- ✅ **Automatic Cleanup**: GPU instance terminates after training

## 🔧 Configuration

### 1. Environment Variables

Copy `.env.gpu.example` to `.env` and update:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_GPU_INSTANCE_TYPE=g4dn.xlarge
AWS_GPU_AMI_ID=ami-00f0918871e41f60d
AWS_KEY_NAME=your-key-pair
AWS_SECURITY_GROUP_ID=sg-xxxxxxxxx
AWS_SUBNET_ID=subnet-xxxxxxxxx
AWS_IAM_INSTANCE_PROFILE=EC2-S3-Access

# S3 Configuration
S3_BUCKET=your-s3-bucket
```

### 2. AWS Permissions

Your IAM role needs these permissions:
- `EC2FullAccess` (to launch/terminate instances)
- `SSMFullAccess` (to run commands on instances)
- `S3FullAccess` (to upload/download training data)

### 3. Security Group

Ensure your security group allows:
- **SSH (22)**: For debugging
- **HTTPS (443)**: For web access
- **Custom TCP (8000)**: For your FastAPI backend

## 🚀 Complete Setup Process

### Phase 1: Instance Setup (One-time)

#### 1. Launch GPU Instance

**Option A: Use Deep Learning AMI (Recommended)**
```bash
# Launch instance with Deep Learning AMI
# AMI ID: ami-00f0918871e41f60d
# Instance Type: g4dn.xlarge or g4dn.2xlarge
# Storage: 30GB+ EBS volume
```

**Option B: Use Ubuntu + Manual Setup**
```bash
# Launch Ubuntu 22.04 LTS
# AMI ID: ami-0bbdd8b17ed981ef9
# Then follow manual setup steps below
```

#### 2. Connect to Instance

```bash
# SSH into your instance
ssh -i "your-key.pem" ubuntu@your-instance-ip
```

#### 3. Install Dependencies (If using Ubuntu)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3-pip python3-dev git

# Install NVIDIA drivers (if not using Deep Learning AMI)
sudo apt install -y nvidia-driver-525
sudo reboot

# After reboot, install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run --silent --toolkit

# Set environment variables
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y libcudnn8

# Install TensorFlow with GPU support
pip3 install tensorflow[and-cuda]==2.13.0
pip3 install boto3 numpy pillow opencv-python scikit-learn requests

# Verify GPU setup
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

#### 4. Clone Your Repository

```bash
# Clone your repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Pull latest changes
git pull origin cursor/greeting-or-placeholder-task-0485

# Install Python dependencies
pip3 install -r ai-training/requirements-gpu.txt
```

#### 5. Configure Environment

```bash
# Copy environment template
cp ai-training/.env.gpu.example ai-training/.env

# Edit configuration
nano ai-training/.env
```

**Required Environment Variables:**
```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_GPU_INSTANCE_TYPE=g4dn.xlarge
AWS_GPU_AMI_ID=ami-00f0918871e41f60d
AWS_KEY_NAME=your-key-pair
AWS_SECURITY_GROUP_ID=sg-xxxxxxxxx
AWS_SUBNET_ID=subnet-xxxxxxxxx
AWS_IAM_INSTANCE_PROFILE=EC2-S3-Access

# S3 Configuration
S3_BUCKET=your-s3-bucket
S3_TRAINING_DATA_PREFIX=training_data
S3_MODELS_PREFIX=models
S3_RESULTS_PREFIX=training_results
S3_SCRIPTS_PREFIX=scripts
```

#### 6. Test the Setup

```bash
# Test GPU training setup
cd ai-training
python test_gpu_training.py
```

### Phase 2: Web Application Setup

#### 1. Start Your Web Application

```bash
# Terminal 1: Start FastAPI backend
cd ai-training
python main.py

# Terminal 2: Start React frontend
npm run dev
```

### Phase 3: Automatic Training (No Manual Steps!)

#### 1. Train Models with GPU

1. **Open web interface**: `http://localhost:3000`
2. **Navigate to**: Signature AI page
3. **Add students** and upload signature images
4. **Click "Train Model"** button
5. **Watch real-time progress** in the web interface

**That's it! The system automatically:**
- ✅ Launches GPU instance
- ✅ Uploads training data to S3
- ✅ Runs training on GPU
- ✅ Streams progress to web interface
- ✅ Saves models to S3
- ✅ Terminates GPU instance

### 3. Training Options

In the web interface, you can choose:
- ✅ **GPU Training**: 10-50x faster (recommended)
- ☁️ **S3 Upload**: Save models to cloud storage
- 📊 **Real-time Progress**: See training metrics live

## 🔍 Monitoring

### Web Interface
- **Progress Bar**: Shows training percentage
- **Real-time Logs**: Training metrics and status
- **Training Results**: Accuracy, loss, and model URLs

### Backend Logs
```bash
# Check training progress
tail -f /var/log/ai-training/training.log

# Check GPU instance status
aws ec2 describe-instances --filters "Name=tag:Purpose,Values=AI-Training"
```

## 🛠️ Troubleshooting

### Common Issues

1. **"GPU not available"**
   - Check AWS credentials
   - Verify IAM permissions
   - Check S3 bucket configuration

2. **"Instance launch failed"**
   - Verify AMI ID is correct
   - Check security group and subnet
   - Ensure key pair exists

3. **"Training failed"**
   - Check instance logs in AWS Console
   - Verify TensorFlow installation
   - Check S3 permissions

### Debug Commands

```bash
# Check GPU availability
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Test AWS connection
aws ec2 describe-instances --max-items 1

# Check S3 access
aws s3 ls s3://your-bucket-name
```

## 📊 Performance

### Training Times (Approximate)
- **CPU Training**: 30-60 minutes
- **GPU Training**: 2-5 minutes
- **Speed Improvement**: 10-50x faster

### Costs
- **g4dn.xlarge**: ~$0.50/hour
- **Typical Training**: $0.05-0.10 per session
- **Auto-cleanup**: No ongoing costs

## 🎯 Next Steps

1. **Test the system**: Try training with a few students
2. **Monitor performance**: Check training times and accuracy
3. **Scale up**: Add more students and training data
4. **Optimize**: Tune hyperparameters for better results

## 📞 Support

If you encounter issues:
1. Check the logs in the web interface
2. Verify AWS configuration
3. Test with a simple training job
4. Check the troubleshooting section above

---

**🎉 Your GPU training system is ready! Click "Train Model" to get started!**