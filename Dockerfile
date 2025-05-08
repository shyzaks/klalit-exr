# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy dependency list to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set an environment variable (optional, for example purposes)
# ENV PYTHONUNBUFFERED=1  (this ensures real-time output, optional)

# Specify the default command to run your app
CMD ["python", "main.py"]
