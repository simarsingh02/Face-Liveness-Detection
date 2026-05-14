FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other ML libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to reduce image size
# Some packages from Windows pip freeze might cause issues on Linux, 
# so we install the core ones explicitly if requirements.txt fails, but try requirements.txt first.
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir tensorflow opencv-python-headless scikit-learn numpy matplotlib joblib scikit-image

# Copy the rest of the application
COPY . .

# Default command to run (can be overridden)
CMD ["python", "inference.py", "--help"]
