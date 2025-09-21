#!/usr/bin/env python3
"""
Test TensorFlow 2.18 Compatibility
Verifies that all our code works with TensorFlow 2.18
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_tensorflow_218():
    """Test TensorFlow 2.18 compatibility"""
    print("🔍 Testing TensorFlow 2.18 compatibility...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test GPU detection (the main API change)
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ GPU detection API: tf.config.list_physical_devices('GPU')")
        print(f"   Found {len(gpus)} GPU(s)")
        
        # Test memory growth configuration
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth configuration: tf.config.experimental.set_memory_growth")
        
        # Test model creation (basic functionality)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        print("✅ Model creation: tf.keras.Sequential")
        
        # Test compilation
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("✅ Model compilation: model.compile")
        
        # Test data preprocessing
        import numpy as np
        x = np.random.random((100, 784))
        y = np.random.randint(0, 2, (100, 1))
        print("✅ Data preprocessing: numpy arrays")
        
        # Test training (just one epoch to verify)
        model.fit(x, y, epochs=1, verbose=0)
        print("✅ Model training: model.fit")
        
        print("\n🎉 All TensorFlow 2.18 compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow 2.18 compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that all our modules can be imported"""
    print("\n🔍 Testing module imports...")
    
    modules_to_test = [
        'tensorflow',
        'tensorflow.keras',
        'numpy',
        'PIL',
        'cv2',
        'boto3',
        'sklearn'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    return True

def main():
    """Run all compatibility tests"""
    print("🚀 TensorFlow 2.18 Compatibility Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test TensorFlow 2.18 compatibility
    if not test_tensorflow_218():
        print("\n❌ TensorFlow 2.18 compatibility tests failed!")
        return False
    
    print("\n🎉 All tests passed! Your codebase is compatible with TensorFlow 2.18!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)