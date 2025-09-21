#!/usr/bin/env python3
"""
Test if tf.config.experimental.set_memory_growth is still valid in TensorFlow 2.18
"""

def test_memory_growth_api():
    """Test the memory growth API"""
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Test the experimental API
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ tf.config.experimental.set_memory_growth: Still works")
        else:
            print("⚠️  No GPUs found, but API should still be valid")
        
        # Check if there's a non-experimental version
        try:
            # This might be the new API in 2.18
            if hasattr(tf.config, 'set_memory_growth'):
                print("✅ tf.config.set_memory_growth: Available (non-experimental)")
            else:
                print("ℹ️  tf.config.set_memory_growth: Not available")
        except:
            pass
            
        return True
    except Exception as e:
        print(f"❌ Memory growth API test failed: {e}")
        return False

if __name__ == "__main__":
    test_memory_growth_api()