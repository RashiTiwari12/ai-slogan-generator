#!/bin/bash
set -e

# Ensure ec2-user owns the app directory and can write logs
sudo chown -R ec2-user:ec2-user /home/ec2-user/slogan
chmod -R 755 /home/ec2-user/slogan

# Stop any existing Streamlit processes
pkill -f streamlit || true  # ignore error if none are running

# Start Streamlit
nohup /home/ec2-user/.local/bin/streamlit run /home/ec2-user/slogan/app.py \
  --server.port 8501 --server.address 0.0.0.0 > /home/ec2-user/slogan/streamlit.log 2>&1 &

