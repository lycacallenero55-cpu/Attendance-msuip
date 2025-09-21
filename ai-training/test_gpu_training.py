#!/usr/bin/env python3
"""
Test GPU Training Setup
Verifies that the GPU training system is properly configured
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import boto3
        print("✅ boto3")
    except ImportError as e:
        print(f"❌ boto3 import failed: {e}")
        return False
    
    try:
        from utils.aws_gpu_training import gpu_training_manager
        print("✅ GPU training manager")
    except ImportError as e:
        print(f"❌ GPU training manager import failed: {e}")
        return False
    
    return True

def test_tensorflow_gpu():
    """Test TensorFlow GPU detection"""
    print("\n🔍 Testing TensorFlow GPU detection...")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("⚠️  No GPUs found (this is normal on CPU-only systems)")
        
        # Test GPU configuration
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU memory growth configured")
            except Exception as e:
                print(f"❌ GPU configuration failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow GPU test failed: {e}")
        return False

def test_aws_config():
    """Test AWS configuration"""
    print("\n🔍 Testing AWS configuration...")
    
    try:
        from utils.aws_gpu_training import gpu_training_manager
        
        # Test AWS credentials
        try:
            gpu_training_manager.ec2_client.describe_instance_types(MaxResults=1)
            print("✅ AWS EC2 credentials working")
        except Exception as e:
            print(f"❌ AWS EC2 credentials failed: {e}")
            return False
        
        # Test S3 access
        try:
            gpu_training_manager.s3_client.list_buckets()
            print("✅ AWS S3 credentials working")
        except Exception as e:
            print(f"❌ AWS S3 credentials failed: {e}")
            return False
        
        # Test SSM access
        try:
            gpu_training_manager.ssm_client.describe_instance_information(MaxResults=1)
            print("✅ AWS SSM credentials working")
        except Exception as e:
            print(f"❌ AWS SSM credentials failed: {e}")
            return False
        
        # Check configuration
        print(f"✅ AMI ID: {gpu_training_manager.ami_id}")
        print(f"✅ Instance Type: {gpu_training_manager.gpu_instance_type}")
        print(f"✅ S3 Bucket: {gpu_training_manager.s3_bucket}")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS configuration test failed: {e}")
        return False

def test_gpu_availability():
    """Test if GPU training is available"""
    print("\n🔍 Testing GPU training availability...")
    
    try:
        from utils.aws_gpu_training import gpu_training_manager
        
        is_available = gpu_training_manager.is_available()
        if is_available:
            print("✅ GPU training is available")
        else:
            print("❌ GPU training is not available")
            print("   Check your AWS configuration and permissions")
        
        return is_available
        
    except Exception as e:
        print(f"❌ GPU availability test failed: {e}")
        return False

def test_training_script():
    """Test that the training script exists and is valid"""
    print("\n🔍 Testing training script...")
    
    script_path = Path(__file__).parent / "scripts" / "train_gpu_tf213.py"
    
    if not script_path.exists():
        print(f"❌ Training script not found: {script_path}")
        return False
    
    print(f"✅ Training script found: {script_path}")
    
    # Test script syntax
    try:
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        compile(script_content, str(script_path), 'exec')
        print("✅ Training script syntax is valid")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Training script syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Training script test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 GPU Training Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("TensorFlow GPU", test_tensorflow_gpu),
        ("AWS Configuration", test_aws_config),
        ("GPU Availability", test_gpu_availability),
        ("Training Script", test_training_script),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! GPU training is ready to use.")
        print("\nNext steps:")
        print("1. Start your web application: python main.py")
        print("2. Open http://localhost:3000")
        print("3. Click 'Train Model' to test GPU training")
    else:
        print("❌ Some tests failed. Please check the configuration.")
        print("\nTroubleshooting:")
        print("1. Check your AWS credentials")
        print("2. Verify IAM permissions")
        print("3. Update .env file with correct values")
        print("4. Check the GPU_TRAINING_SETUP.md guide")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)