# Stage 1: Base Image
FROM python:3.10-slim

# Set up the working directory
WORKDIR /app

# Create an `app/ui/requirements.txt` file with:
# streamlit
# requests
# PyYAML

# Copy the UI-specific requirements file
COPY app/ui/requirements.txt ./ui_requirements.txt
COPY core/requirements.txt ./core_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r ui_requirements.txt
RUN pip install --no-cache-dir -r core_requirements.txt

# Copy the UI and core application code
COPY app/ui/ /app/app/ui/
COPY core/ /app/core/
COPY config.yaml /app/config.yaml

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
# We need to bind to 0.0.0.0 and disable headless mode for it to work in a container
CMD ["streamlit", "run", "app/ui/search_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
