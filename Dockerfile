FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start FastAPI on Hugging Face required port
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
