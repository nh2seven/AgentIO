#!/bin/bash

# Define the project root and data dirs
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

# Create subdirectories`
mkdir -p "$DATA_DIR"/{docs,faiss,logs}
