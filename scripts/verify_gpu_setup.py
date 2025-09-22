#!/usr/bin/env python3
"""
GPU Setup Verification Script
Verifies that the AI training system is properly configured for GPU training
"""

import sys
import os
import json
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Incompatible (requires 3.12+)")
        return False

def check_tensorflow():
    """Check TensorFlow installation and GPU support"""
    print("\n🔍 Checking TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} installed")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
            return True
        else:
            print("❌ No GPU detected - TensorFlow will use CPU")
            return False
    except ImportError as e:
        print(f"❌ TensorFlow not installed: {e}")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("\n🔍 Checking dependencies...")
    required_packages = [
        'numpy', 'pillow', 'opencv-python', 'boto3', 'supabase',
        'fastapi', 'uvicorn', 'starlette', 'pydantic'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("✅ All dependencies installed")
        return True

def check_project_structure():
    """Check if project structure is correct"""
    print("\n🔍 Checking project structure...")
    required_dirs = [
        'models', 'api', 'utils', 'scripts'
    ]
    
    required_files = [
        'models/global_signature_classifier.py',
        'api/training.py',
        'api/verification.py',
        'main.py',
        'requirements-gpu.txt'
    ]
    
    missing_items = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            missing_items.append(dir_path)
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            missing_items.append(file_path)
    
    if missing_items:
        print(f"\n❌ Missing items: {', '.join(missing_items)}")
        return False
    else:
        print("✅ Project structure is correct")
        return True

def check_environment_variables():
    """Check if environment variables are set"""
    print("\n🔍 Checking environment variables...")
    required_vars = [
        'AWS_REGION', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
        'S3_BUCKET', 'AWS_GPU_EXISTING_INSTANCE_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}")
        else:
            print(f"❌ {var} - Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print("✅ All environment variables are set")
        return True

def check_gpu_drivers():
    """Check if GPU drivers are installed"""
    print("\n🔍 Checking GPU drivers...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA drivers installed")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA drivers not installed or not working")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - GPU drivers not installed")
        return False

def test_global_classifier():
    """Test the global signature classifier"""
    print("\n🔍 Testing Global Signature Classifier...")
    try:
        from models.global_signature_classifier import GlobalSignatureClassifier
        
        # Test initialization
        classifier = GlobalSignatureClassifier(max_students=10)
        print("✅ GlobalSignatureClassifier initialized successfully")
        
        # Test model creation
        model = classifier.create_model(num_classes=5)
        print(f"✅ Model created with {model.count_params()} parameters")
        
        return True
    except Exception as e:
        print(f"❌ GlobalSignatureClassifier test failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 GPU Setup Verification Script")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("TensorFlow", check_tensorflow),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Environment Variables", check_environment_variables),
        ("GPU Drivers", check_gpu_drivers),
        ("Global Classifier", test_global_classifier)
    ]
    
    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION RESULTS")
    print("=" * 50)
    
    passed = 0
    total = len(checks)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your GPU setup is ready for production!")
        return 0
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
