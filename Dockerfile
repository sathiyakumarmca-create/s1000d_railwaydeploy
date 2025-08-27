# Use the official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install all the required Python dependencies
# The --no-cache-dir flag helps keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other project files into the container
# This includes app.py, s1000d_retriever_agent.py, etc.
COPY . .

# Expose the port that Streamlit uses. Railway will automatically
# map this to a public port.
EXPOSE 8501

# Command to run the Streamlit application
# The --server.port 8501 and --server.address 0.0.0.0 are critical
# for Streamlit to be accessible from outside the container.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
