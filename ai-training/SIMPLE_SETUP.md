# 🚀 Simple Setup Guide - Signature Verification AI System

## 🎯 **ONE Clear Solution - No More Confusion!**

### **For Your Windows PC (Python 3.10.11):**
```bash
pip install -r requirements-local.txt
```

### **For AWS GPU Instance (Python 3.12.6):**
```bash
pip install -r requirements-gpu.txt
```

## ✅ **That's It!**

**Both use TensorFlow 2.13.1** - proven stable and compatible with your systems.

## 🔧 **If You Get Errors:**

**1. Clear your pip cache:**
```bash
pip cache purge
```

**2. Create a fresh virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-local.txt
```

**3. If still having issues, install one by one:**
```bash
pip install tensorflow==2.13.1
pip install numpy==1.24.3
pip install scikit-learn==1.3.2
pip install scipy==1.10.1
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

## 🎉 **Ready to Go!**

Once installed:
1. **Start backend**: `python main.py`
2. **Start frontend**: `npm run dev`
3. **Upload signatures**: Use the web interface
4. **Train models**: Click "Train Model" button

**Your capstone project is ready!** 🎓