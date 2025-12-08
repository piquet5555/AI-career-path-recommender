# Start from a standard, stable Python base image
FROM python:3.11-slim

# Set environment variables for the application
ENV PYTHONUNBUFFERED 1
ENV PORT 8000

# Install system dependencies needed for libraries like spacy and cryptography
# 'build-essential' is the key fix for compilation errors
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies FIRST
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app

# The command to run the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT