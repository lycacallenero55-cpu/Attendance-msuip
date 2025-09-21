# 🔧 Installation Guide - Fix Dependency Conflicts

## 🚨 **If you get dependency conflicts, try these solutions in order:**

### **Solution 1: Use the Safe Version (Recommended)**
```bash
# Use this if you get numpy conflicts
pip install -r requirements-local-safe.txt
```

### **Solution 2: Install TensorFlow First**
```bash
# Install TensorFlow first, then let it resolve dependencies
pip install tensorflow==2.18.0
pip install -r requirements-local.txt
```

### **Solution 3: Use pip-tools (Advanced)**
```bash
# Install pip-tools
pip install pip-tools

# Create a constraints file
echo "tensorflow==2.18.0" > constraints.txt
echo "numpy>=1.26.0,<2.1.0" >> constraints.txt

# Install with constraints
pip install -r requirements-local.txt -c constraints.txt
```

### **Solution 4: Manual Installation (If all else fails)**
```bash
# Install packages one by one
pip install tensorflow==2.18.0
pip install "numpy>=1.26.0,<2.1.0"
pip install "scikit-learn>=1.3.0"
pip install "scipy>=1.10.0"
pip install pillow==10.2.0
pip install opencv-python==4.9.0.80
pip install fastapi==0.92.0
pip install uvicorn[standard]==0.22.0
pip install python-multipart==0.0.9
pip install boto3==1.34.34
pip install supabase>=2.6.0
pip install requests==2.31.0
pip install python-dotenv==1.0.1
pip install orjson==3.9.15
pip install pydantic==1.10.13
pip install aiohttp>=3.8,<4
pip install psutil
```

## 🎯 **Recommended Approach**

**For your Windows PC with Python 3.10.11:**

1. **Try Solution 1 first** (safest):
   ```bash
   pip install -r requirements-local-safe.txt
   ```

2. **If that works, you're done!** ✅

3. **If you still get conflicts, try Solution 2**:
   ```bash
   pip install tensorflow==2.18.0
   pip install -r requirements-local.txt
   ```

## 🔍 **Why This Happens**

- **TensorFlow 2.18** requires `numpy>=1.26.0`
- **Some packages** expect older numpy versions
- **Python 3.10.11** has specific compatibility requirements
- **pip dependency resolver** sometimes can't find a solution

## ✅ **Verification**

After installation, test with:
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('NumPy version:', tf.__version__)"
```

**Expected output:**
```
TensorFlow version: 2.18.0 (or 2.13.1)
NumPy version: 1.26.x (or 1.24.3)
```

## 🚀 **Next Steps**

Once installed successfully:
1. **Start the backend**: `python main.py`
2. **Start the frontend**: `npm run dev`
3. **Test the system**: Upload some signature images
4. **Train a model**: Use the web interface

**Your system will work perfectly for your capstone project!** 🎓