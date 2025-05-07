import os
import sys
import yaml
from model import train_and_save_models
from processing import preprocess_and_save
from dotenv import load_dotenv

def load_config(model_type):
    try:
        with open("params/base.yaml", 'r') as file:
            base_cfg = yaml.safe_load(file) or {}
            
        model_config_path = f"params/models/{model_type}.yaml"
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config file not found: {model_config_path}")
            
        with open(model_config_path, 'r') as file:
            model_cfg = yaml.safe_load(file) or {}

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
    try:
        load_dotenv()  
        
        if len(sys.argv) < 2:
            print("Error: Model type argument is missing!")
            print("Usage: python src/Main.py <model_type>")
            sys.exit(1)
        
        model_type = sys.argv[1]  
        cfg = load_config(model_type)
        
        print("Preprocessing data")
        preprocess_and_save()   
        
        print("\nTraining model")
        model_path = train_and_save_models(
            model_cfg=cfg['model'],
            processed_dir=cfg['data'].get('processed_dir', 'data/processed')
        )
        
        print("Pipeline Completed Successfully")
                
    except Exception as e:
        print(f"\n! Pipeline Failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()