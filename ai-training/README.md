# Signature AI Training Backend

AI-powered signature verification training backend using TensorFlow and Supabase.

## Features

- **Signature Verification Training**: Train Siamese neural networks for signature verification
- **Real-time Verification**: Verify signatures against trained models
- **Supabase Integration**: Store models and metadata in Supabase
- **RESTful API**: FastAPI-based API for easy integration
- **Image Processing**: Automatic image validation and preprocessing

## Setup

### 1. Install Python 3.10+
```bash
# Python 3.10+ is required (3.10.11 works perfectly)
# Windows: Download from python.org
# macOS: brew install python@3.10
# Ubuntu: sudo apt install python3.10 python3.10-venv
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Supabase credentials
```

### 5. Run the Server
```bash
python main.py
```

## API Endpoints

### Training
- `POST /api/training/start` - Start training a model
- `GET /api/training/status/{model_id}` - Get training status
- `GET /api/training/models` - Get all trained models

### Verification
- `POST /api/verification/verify` - Verify a signature
- `GET /api/verification/models/{student_id}` - Get available models

## Usage

### Training a Model
```python
import requests

# Upload images and start training
files = {
    'genuine_files': [open('genuine1.jpg', 'rb'), open('genuine2.jpg', 'rb')],
    'forged_files': [open('forged1.jpg', 'rb')]
}
data = {'student_id': 123}

response = requests.post('http://localhost:8000/api/training/start', files=files, data=data)
```

### Verifying a Signature
```python
# Verify a signature
files = {
    'reference_files': [open('reference.jpg', 'rb')],
    'test_file': open('test_signature.jpg', 'rb')
}
data = {'model_id': 'your-model-id'}

response = requests.post('http://localhost:8000/api/verification/verify', files=files, data=data)
```

## Configuration

Edit `.env` file to configure:
- Supabase credentials
- Model parameters
- Training settings
- Storage options

## Requirements

- Python 3.10+ (3.10.11 works perfectly)
- TensorFlow 2.15+
- Supabase account
- Minimum 4GB RAM
- CPU with AVX support (recommended)