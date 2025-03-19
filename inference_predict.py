# Utils
from models.classification_models import HybridVisionClassification
from config import Config
from utils.results_util import get_results_folder
from utils.logger import get_logger

# PyTorch
import torch

# Training & Inference
from utils import inference_utils



def main(mode):
    ##upload configs
    config = Config(mode)
    
    #create results path
    config.add_curr_run_folder(get_results_folder(config.paths.results_path))
    
    # Init logger
    config.logger = get_logger(mode, config.curr_run_results_path)
    config.logger.info(f"Running {mode} of {config.test} for {config.model_type_str}")
    config.logger.info(F"\nCurrent run number is {config.curr_run_results_folder}")
    
    #Load model and weights of each fold
    fold_models = [HybridVisionClassification(config,1).to(config.device) for _ in range(len(config.weights_paths))]
    for idx, (model, weight_path) in enumerate(zip(fold_models, config.weights_paths)):
        config.logger.info(f"Loading weights for fold {idx+1}")
        model.load_state_dict(torch.load(weight_path, map_location=config.device))
        
    # run inference
    inference_utils.main(config.test,fold_models,config,mode)
        
if __name__ == "__main__":
    mode = 'inference'
    main(mode)

