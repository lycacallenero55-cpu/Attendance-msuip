import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")
    
    # AWS / S3 Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL")
    S3_USE_PRESIGNED_GET = os.getenv("S3_USE_PRESIGNED_GET", "true").lower() == "true"
    
    # GPU Training Configuration
    AWS_KEY_NAME = os.getenv("AWS_KEY_NAME")
    AWS_SECURITY_GROUP_ID = os.getenv("AWS_SECURITY_GROUP_ID")
    AWS_SUBNET_ID = os.getenv("AWS_SUBNET_ID")
    AWS_GPU_EXISTING_INSTANCE_ID = os.getenv("AWS_GPU_EXISTING_INSTANCE_ID")
    MAX_CONCURRENT_GPU_INSTANCES = int(os.getenv("MAX_CONCURRENT_GPU_INSTANCES", "1"))
    AWS_GPU_AMI_ID = os.getenv("AWS_GPU_AMI_ID")
    AWS_GPU_GITHUB_REPO = os.getenv("AWS_GPU_GITHUB_REPO")
    AWS_IAM_INSTANCE_PROFILE = os.getenv("AWS_IAM_INSTANCE_PROFILE")
    AWS_GPU_INSTANCE_TYPE = os.getenv("AWS_GPU_INSTANCE_TYPE")
    USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "false").lower() == "true"
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model Configuration
    MODEL_IMAGE_SIZE = int(os.getenv("MODEL_IMAGE_SIZE", "224"))
    MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE", "32"))
    MODEL_EPOCHS = int(os.getenv("MODEL_EPOCHS", "50"))
    MODEL_LEARNING_RATE = float(os.getenv("MODEL_LEARNING_RATE", "0.001"))
    
    # Storage Configuration
    LOCAL_MODELS_DIR = os.getenv("LOCAL_MODELS_DIR", "./models")
    
    # Training Configuration
    MIN_GENUINE_SAMPLES = int(os.getenv("MIN_GENUINE_SAMPLES", "10"))
    MIN_FORGED_SAMPLES = int(os.getenv("MIN_FORGED_SAMPLES", "5"))
    MAX_TRAINING_TIME = int(os.getenv("MAX_TRAINING_TIME", "3600"))

# Create a global settings instance
settings = Settings()
