import os
import sys
import yaml
from model import train_and_save_models
from processing import preprocess_and_save

def load_config(model_type):
    """Load and merge configuration files with error handling."""
    try:
        # Load base configuration
        with open("params/base.yaml", 'r') as file:
            base_cfg = yaml.safe_load(file) or {}
            
        # Load model-specific configuration
        model_config_path = f"params/models/{model_type}.yaml"
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config file not found: {model_config_path}")
            
        with open(model_config_path, 'r') as file:
            model_cfg = yaml.safe_load(file) or {}

        # Set default output directory if not specified
        base_cfg.setdefault('model', {}).setdefault('output_dir', 'models')
        
        return {
            'data': base_cfg.get('data', {}),
            'model': {
                **model_cfg,
                'output_dir': base_cfg['model']['output_dir']
            }
        }
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {str(e)}")

def main():
    """Main execution function with error handling."""
    try:
        # Check if model argument is passed
        if len(sys.argv) < 2:
            print("Error: Model type argument is missing!")
            print("Usage: python src/Main.py <model_type>")
            sys.exit(1)
        
        model_type = sys.argv[1]  # Get model type from command-line argument
        
        print(f"Starting pipeline for model: {model_type}")
        
        # Load config
        print("Loading configuration...")
        cfg = load_config(model_type)
        
        # Step 1: Preprocessing
        print("\nStep 1: Running preprocessing...")
        preprocess_and_save()
        
        # Step 2: Model training
        print("\nStep 2: Training model...")
        train_and_save_models(
            model_cfg=cfg['model'],
            processed_dir=cfg['data'].get('processed_dir', 'data/processed')
        )
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        # Optionally exit with error code
        sys.exit(1)

if __name__ == "__main__":
    main()
