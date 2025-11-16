# Set working directory

# Copy requirements and install Drake packages
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /workspace/

# Install air_hockey_challenge package in editable mode
RUN cd /workspace/air_hockey_challenge && pip install --no-cache-dir -e .

# Set PYTHONPATH to include workspace
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# Default command (can be overridden)
CMD ["python", "scripts/drake_implementation.py"]
