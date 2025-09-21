#!/usr/bin/env python3
"""
Comprehensive TensorFlow 2.18 Compatibility Test
Tests all TensorFlow APIs used in the ai-training codebase
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_tensorflow_imports():
    """Test TensorFlow imports"""
    print("🔍 Testing TensorFlow imports...")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        print(f"✅ TensorFlow version: {tf.__version__}")
        print("✅ Keras imports successful")
        return True
    except Exception as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False

def test_gpu_detection():
    """Test GPU detection APIs"""
    print("\n🔍 Testing GPU detection APIs...")
    
    try:
        import tensorflow as tf
        
        # Test the main API used in the codebase
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ tf.config.list_physical_devices('GPU'): {len(gpus)} GPU(s)")
        
        # Test memory growth configuration
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ tf.config.experimental.set_memory_growth: Working")
        
        return True
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False

def test_model_creation():
    """Test model creation APIs"""
    print("\n🔍 Testing model creation APIs...")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Test Sequential model creation (used in tfdata.py)
        model = keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=(784,)),
            layers.Dense(1, activation='sigmoid')
        ])
        print("✅ keras.Sequential: Working")
        
        # Test functional API (used in signature_embedding_model.py)
        inputs = keras.Input(shape=(224, 224, 3))
        x = layers.Conv2D(32, 3, activation='relu')(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10)(x)
        model = keras.Model(inputs, outputs)
        print("✅ keras.Model functional API: Working")
        
        # Test MobileNetV2 (used in signature_embedding_model.py)
        base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        print("✅ keras.applications.MobileNetV2: Working")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_optimizers_and_metrics():
    """Test optimizers and metrics APIs"""
    print("\n🔍 Testing optimizers and metrics APIs...")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        # Test AdamW optimizer (used in signature_embedding_model.py)
        optimizer = keras.optimizers.AdamW(learning_rate=0.001)
        print("✅ keras.optimizers.AdamW: Working")
        
        # Test TopKCategoricalAccuracy (used in signature_embedding_model.py)
        metric = keras.metrics.TopKCategoricalAccuracy(k=3)
        print("✅ keras.metrics.TopKCategoricalAccuracy: Working")
        
        # Test to_categorical (used in signature_embedding_model.py)
        import numpy as np
        y = np.array([0, 1, 2, 1])
        y_cat = tf.keras.utils.to_categorical(y, num_classes=3)
        print("✅ tf.keras.utils.to_categorical: Working")
        
        return True
    except Exception as e:
        print(f"❌ Optimizers and metrics failed: {e}")
        return False

def test_data_processing():
    """Test data processing APIs"""
    print("\n🔍 Testing data processing APIs...")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np
        
        # Test tf.data.Dataset (used in tfdata.py)
        x = np.random.random((100, 224, 224, 3))
        y = np.random.randint(0, 2, (100, 1))
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        print("✅ tf.data.Dataset.from_tensor_slices: Working")
        
        # Test tf.data.AUTOTUNE (used in tfdata.py)
        autotune = tf.data.AUTOTUNE
        print("✅ tf.data.AUTOTUNE: Working")
        
        # Test tf.image.resize (used in signature_embedding_model.py)
        image = tf.random.normal((224, 224, 3))
        resized = tf.image.resize(image, [112, 112])
        print("✅ tf.image.resize: Working")
        
        # Test tf.math operations (used in tfdata.py)
        num = tf.constant(100)
        val_size = tf.cast(tf.math.round(tf.cast(num, tf.float32) * 0.2), tf.int32)
        print("✅ tf.math operations: Working")
        
        return True
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def test_training_workflow():
    """Test complete training workflow"""
    print("\n🔍 Testing complete training workflow...")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        import numpy as np
        
        # Create a simple model
        model = keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=(784,)),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy data
        x = np.random.random((100, 784))
        y = np.random.randint(0, 2, (100, 1))
        
        # Test training
        history = model.fit(x, y, epochs=1, verbose=0)
        print("✅ Model training: Working")
        
        # Test prediction
        predictions = model.predict(x[:10], verbose=0)
        print("✅ Model prediction: Working")
        
        return True
    except Exception as e:
        print(f"❌ Training workflow failed: {e}")
        return False

def test_model_imports():
    """Test importing the actual model classes"""
    print("\n🔍 Testing model class imports...")
    
    try:
        from models.signature_embedding_model import SignatureEmbeddingModel
        print("✅ SignatureEmbeddingModel import: Working")
        
        from models.global_signature_classifier import GlobalSignatureClassifier
        print("✅ GlobalSignatureClassifier import: Working")
        
        from utils.tfdata import build_preprocess_layers, make_tfdata_from_numpy
        print("✅ tfdata utilities import: Working")
        
        from utils.signature_preprocessing import SignaturePreprocessor
        print("✅ SignaturePreprocessor import: Working")
        
        return True
    except Exception as e:
        print(f"❌ Model imports failed: {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("🚀 TensorFlow 2.18 Comprehensive Compatibility Test")
    print("=" * 60)
    
    tests = [
        test_tensorflow_imports,
        test_gpu_detection,
        test_model_creation,
        test_optimizers_and_metrics,
        test_data_processing,
        test_training_workflow,
        test_model_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The codebase is fully compatible with TensorFlow 2.18!")
        return True
    else:
        print("❌ Some tests failed. The codebase may have compatibility issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)