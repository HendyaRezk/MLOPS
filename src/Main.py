import hydra
from omegaconf import DictConfig
from processing import preprocess_and_save
from model import train_and_save_models

@hydra.main(config_path="../config", config_name="config", version_base=None)
def run_pipeline(cfg: DictConfig):
    print("\nStep 1: Preprocessing...")
    preprocess_and_save(cfg.data)

    print("\nStep 2: Training model...")
    train_and_save_models(cfg.model, cfg.data.processed_dir)

if __name__ == "__main__":
    run_pipeline()
