# Use a lightweight Python image as base
FROM python:3.9-slim

# Set environment variables to avoid writing pyc files and to ensure logs are visible
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose port 5000 (if your app listens on this port)
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
