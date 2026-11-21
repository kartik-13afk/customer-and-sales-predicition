import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IS_VERCEL = bool(os.environ.get('VERCEL'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mca-customer-segmentation-secret-key-2026')
    BASE_DIR = BASE_DIR
    IS_VERCEL = IS_VERCEL
    
    if IS_VERCEL:
        DATA_RAW_DIR = os.path.join('/tmp', 'data', 'raw')
        DATA_PROCESSED_DIR = os.path.join('/tmp', 'data', 'processed')
        MODELS_DIR = os.path.join('/tmp', 'models')
        DATABASE_PATH = os.path.join('/tmp', 'database', 'app.db')
    else:
        DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
        DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
        MODELS_DIR = os.path.join(BASE_DIR, 'models')
        DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'app.db')
        
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB maximum upload
    ALLOWED_EXTENSIONS = {'csv'}

config = Config()
