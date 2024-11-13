time accelerate launch --config_file accelerate_single_config.yaml scripts/train.py
time accelerate launch --config_file accelerate_single_config.yaml scripts/train.py --config config/dgx.py:compressibility
time accelerate launch --config_file accelerate_single_config.yaml scripts/train.py --config config/dgx.py:aesthetic
time accelerate launch --config_file accelerate_single_config.yaml scripts/train.py --config config/dgx.py:prompt_image_alignment
time accelerate launch --config_file accelerate_single_config.yaml scripts/train.py --config config/dgx.py:prompt_pickscore

# for development
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/dgx.py:aesthetic_test