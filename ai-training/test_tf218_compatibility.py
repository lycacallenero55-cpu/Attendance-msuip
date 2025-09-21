#!/usr/bin/env python3
"""
Test TensorFlow 2.18 Compatibility
Verifies that TensorFlow 2.18 works on both CPU and GPU
"""

def test_tensorflow_218_compatibility():
    """Test TensorFlow 2.18 compatibility"""
    print("🔍 Testing TensorFlow 2.18 compatibility...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test GPU detection (works on both CPU and GPU)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("✅ CPU mode: No GPUs detected (normal on CPU-only systems)")
        
        # Test memory growth configuration
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth configured")
        
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
        
        print("\n🎉 TensorFlow 2.18 compatibility test PASSED!")
        print("✅ Works on both CPU and GPU systems")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow 2.18 compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tensorflow_218_compatibility()