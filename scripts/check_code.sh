#!/bin/bash
set -e

echo "Running Black..."
black src/britekit tests

echo "Running Ruff..."
ruff check src/britekit tests

echo "Running Mypy..."
mypy src/britekit tests
