# Use official Python runtime as a parent image
FROM python:3.9-slim

# Prevents Python from writing .pyc files to disk and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY Speech-To-Text-Gen-AI/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
  pip install -r requirements.txt

# Copy project code
COPY Speech-To-Text-Gen-AI /app

# Expose application port (matches service targetPort)
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
