.PHONY: setup download prepare train inference upload clean

PYTHON := python
DATA_DIR := data/processed/mcv-scripted-mn-v24.0/cv-corpus-24.0-2025-12-05/mn
OUTPUT_DIR := data/prepared
CHECKPOINT_DIR := checkpoints/mongolian_vits2
CONFIG := configs/mongolian.yaml

setup:
	pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_dataset.py --output-dir data/raw

prepare:
	$(PYTHON) scripts/prepare_dataset.py \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR)

prepare-fast:
	$(PYTHON) scripts/prepare_dataset.py \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--no-denoise

train:
	$(PYTHON) scripts/train.py \
		--config ../../$(CONFIG) \
		--model-dir ../../$(CHECKPOINT_DIR)

inference:
	@if [ -z "$(TEXT)" ]; then \
		echo "Usage: make inference TEXT='Your text here'"; \
		exit 1; \
	fi
	$(PYTHON) scripts/inference.py \
		--text "$(TEXT)" \
		--checkpoint $(CHECKPOINT_DIR)/G_latest.pth \
		--config $(CONFIG) \
		--output output.wav

upload:
	$(PYTHON) scripts/upload_to_huggingface.py \
		--model-dir $(CHECKPOINT_DIR) \
		--create-card

clean:
	rm -rf data/prepared/clips/temp_wav
	rm -rf data/prepared/clips/denoised
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf data/prepared
	rm -rf $(CHECKPOINT_DIR)
