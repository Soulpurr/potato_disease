# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy your Flask API code into the container at /app
COPY app.py /app/

# Copy your model directory into the container at /app/model
COPY model/ /app/model/

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any needed dependencies specified in requirements.txt
RUN pip install tensorflow==2.10.0 protobuf==3.20 numpy==1.21.5 Flask Flask-Cors

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
