# Adaptive Intersection AI Copilot - Build Automation
# Based on lessons learned from PROJECT_HISTORY.md

# Configuration
PYTHON = python3
VENV_DIR = venv
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip

# SUMO paths (will be set after venv creation)
SUMO_BIN = $(VENV_DIR)/bin
SUMO_TOOLS = $(VENV_DIR)/lib/python3.9/site-packages/sumo/tools
SUMO_EXEC = $(VENV_DIR)/bin/sumo

# Project paths
SIM_DIR = sim
DASHBOARD_DIR = dashboard
RESULTS_DIR = results
SRC_DIR = src

.PHONY: help setup clean test sim-fixed sim-rl dashboard install-deps

# Default target
help:
	@echo "Adaptive Intersection AI Copilot - Available Commands:"
	@echo "  setup        - Set up virtual environment and install dependencies"
	@echo "  install-deps - Install Python dependencies"
	@echo "  sim-fixed    - Run fixed-time traffic controller simulation"
	@echo "  sim-rl       - Run RL-based traffic controller simulation"
	@echo "  dashboard    - Launch Streamlit dashboard"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean up generated files"
	@echo "  help         - Show this help message"

# Set up virtual environment and install dependencies
setup: $(VENV_DIR)
	@echo "Setting up virtual environment..."
	$(VENV_PIP) install -r requirements.txt
	@echo "Setup complete! Run 'make sim-fixed' to test the simulation."

# Create virtual environment
$(VENV_DIR):
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip

# Install dependencies
install-deps: $(VENV_DIR)
	$(VENV_PIP) install -r requirements.txt

# Run simple simulation
sim-simple: $(VENV_DIR)
	@echo "Running simple traffic simulation..."
	$(VENV_PYTHON) $(SIM_DIR)/run_simple.py

# Run fixed-time simulation
sim-fixed: $(VENV_DIR)
	@echo "Running fixed-time traffic controller simulation..."
	$(VENV_PYTHON) $(SIM_DIR)/run_fixed.py

# Run RL-based simulation
sim-rl: $(VENV_DIR)
	@echo "Running RL-based traffic controller simulation..."
	$(VENV_PYTHON) $(SIM_DIR)/run_rl.py

# Run CV-integrated simulation
sim-cv: $(VENV_DIR)
	@echo "Running CV-integrated traffic simulation..."
	$(VENV_PYTHON) $(SIM_DIR)/run_cv_integrated.py

# Quick CV test
test-cv: $(VENV_DIR)
	@echo "Running quick CV integration test..."
	$(VENV_PYTHON) $(SIM_DIR)/test_cv_quick.py

# AI traffic control demo
ai-control: $(VENV_DIR)
	@echo "Running AI traffic control demo..."
	$(VENV_PYTHON) $(SIM_DIR)/run_ai_traffic_control.py

# Launch dashboard
dashboard: $(VENV_DIR)
	@echo "Launching Streamlit dashboard..."
	$(VENV_PYTHON) -m streamlit run $(DASHBOARD_DIR)/app.py

# Run tests
test: $(VENV_DIR)
	$(VENV_PYTHON) -m pytest $(SRC_DIR)/tests/ -v

# Clean up generated files
clean:
	rm -rf $(RESULTS_DIR)/*
	rm -rf __pycache__/
	rm -rf $(SRC_DIR)/**/__pycache__/
	rm -rf $(SIM_DIR)/**/__pycache__/
	rm -rf $(DASHBOARD_DIR)/**/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Clean everything including venv
clean-all: clean
	rm -rf $(VENV_DIR)

# Check SUMO installation
check-sumo: $(VENV_DIR)
	@echo "Checking SUMO installation..."
	$(VENV_PYTHON) -c "import sumolib; print('SUMO imported successfully')"
	$(VENV_PYTHON) -c "import traci; print('TraCI imported successfully')"
	$(VENV_PYTHON) -c "import subprocess; result = subprocess.run(['$(SUMO_EXEC)', '--version'], capture_output=True, text=True); print('SUMO version:', result.stdout.strip())"