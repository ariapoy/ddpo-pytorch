time accelerate launch --config_file accelerate_single_config.yaml scripts/train.py
time accelerate launch --config_file accelerate_single_config.yaml scripts/train.py --config config/dgx.py:compressibility
time accelerate launch --config_file accelerate_single_config.yaml scripts/train.py --config config/dgx.py:aesthetic

# for development
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/dgx.py:aesthetic_test