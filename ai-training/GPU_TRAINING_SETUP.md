# 🚀 GPU Training Setup Guide

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

## 🚀 How to Use

### 1. Start Your Web Application

```bash
# Terminal 1: Start FastAPI backend
cd ai-training
python main.py

# Terminal 2: Start React frontend
npm run dev
```

### 2. Train Models with GPU

1. **Open web interface**: `http://localhost:3000`
2. **Navigate to**: Signature AI page
3. **Add students** and upload signature images
4. **Click "Train Model"** button
5. **Watch real-time progress** in the web interface

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