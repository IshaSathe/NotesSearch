# Python parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy the Hugging Face model into the container
COPY all-MiniLM-L6-v2 /app/all-MiniLM-L6-v2

# Copy the entire app directory
COPY . /app/

# Expose the app's port
EXPOSE 5000

# run app
CMD ["python", "app.py"]
