
import json
import os
import platform
import torch
import sys
import sys
import os
import glob


class Config:
    def __init__(self, mode):
        """
        Initialize configuration based on mode (train/inference).
        """
        self.mode = mode.lower()
        
        # Load common config
        # Prepare paths
        self.main_config_path = f"configs/{self.mode}_config.json"
        self.load_main_config(self.main_config_path)
        
        # Load processing config (common for both)
        self.processing_config_path = "configs/processing_config.json"
        self.load_processing_config(self.processing_config_path)
        
        #load paths
        self.paths = self.SystemPaths(self.system_paths,mode)
        
        # Set device
        self.set_device()

        # Mode-specific updates
        if self.mode == "train":
            self.setup_training()
        elif self.mode in ["inference","predict"]:
            self.setup_inference()
            
    
        # Number of input c
        self.image_in_channels = len(self.input_images)
        
        #Dataset types
        self.oof = 'oof' ##out of fold (validation) data
        self.train = 'train' #train
        self.test = 'test' #test
        
        #oof default thershold 
        self.oof_threshold = 0.5
        
    
    def load_main_config(self, config_path):
        """Load the main configuration file."""
        with open(config_path, 'r') as file:
            self.main_config = json.load(file)
            
        self.model_type_str = self.main_config.get("model", "")
        self.model_type = self.parse_model_type(self.model_type_str)
        self.device_param = self.main_config.get("device", {})
        self.system_paths = self.main_config.get("system_paths", {})
        self.csv_columns = self.CSVColumns(self.main_config.get("csv_columns", {}))
        self.dataset_params = self.DatasetParams(self.main_config.get("dataset_params",{}),self.mode)
        self.model_params = self.ModelParams(self.main_config.get("model_params", {}),self.mode)
        self.global_params = self.GlobalParams(self.main_config.get("global_params", {}),self.mode)
        
    def load_processing_config(self, processing_config_path):
        """Load the processing configuration file."""
        with open(processing_config_path, 'r') as file:
            self.processing_config = json.load(file)
        self.label_encoding = self.processing_config.get('label_encoding', {})

    def parse_model_type(self, model_type):
        """Convert model type to a list if necessary."""
        
        return model_type.split('_') if '_' in model_type else [model_type]

    def set_device(self):
        """Set the computation device (GPU/CPU)."""
        self.device = torch.device(self.device_param.get('gpu', 'cuda') if torch.cuda.is_available() else 'cpu')
        print('Device available now:', self.device)

    def setup_training(self):
        """Set up training-specific configurations."""
        self.input_images = self.main_config.get("input_images", [])
        


    def setup_inference(self):        
        # Load model configurations for inference
        with open("configs/trained_weights_config.json", 'r') as model_config_file:
            self.model_config = json.load(model_config_file)
        self.optimal_threshold = self.model_config[self.model_type_str].get("optimal_ths", 0.5)
        self.weights_folder = os.path.join('weights',self.model_type_str)
        self.weights_paths =  [os.path.join(self.weights_folder,file) for file in glob.glob1(self.weights_folder,'*pth')]
        self.input_images = self.model_config[self.model_type_str].get("input_images", [])
        self.dataset_params.resize_height = self.model_config[self.model_type_str].get("resize_height", 224)
        self.dataset_params.resize_width = self.model_config[self.model_type_str].get("resize_width", 224)
        
    def results_prediction_path(self):
        self.preds_results_path = {}
        #Save predictions for 'inference' and 'predict'
        if self.mode != 'train':
            dataset_type = 'test'
            self.preds_results_path = os.path.join('results',self.mode,self.curr_run_results_folder,f"{dataset_type}_predictions.csv")
        
    def add_curr_run_folder(self,curr_run_folder):
        self.curr_run_results_folder = curr_run_folder
        self.curr_run_results_path = os.path.join(self.paths.results_path,self.curr_run_results_folder)
        os.makedirs(self.curr_run_results_path)
        
        self.results_prediction_path()
        
        
    class DatasetParams:
        def __init__(self, params,mode):
            """Initialize dataset parameters from a dictionary."""
            self.dict = params
            # global params
            self.pixel_around_tumor = params["pixel_around_tumor"]
           
            # for train
            if mode == 'train':
                self.resize_height = params["resize_height"]
                self.resize_width = params["resize_width"]
                self.vertical_flip = params["vertical_flip"]
                self.horizontal_flip = params["horizontal_flip"]
                self.rotate_limit = params["rotate_limit"]
                self.scale_limit = params["scale_limit"]
                self.shift_limit = params["shift_limit"]
                self.slice_thick = params["slice_thick"]
                
    
    class ModelParams:
        def __init__(self, params,mode):
            """Initialize trained model parameters from a dictionary."""
            self.dict = params
            self.test_batch_size = params.get("test_batch_size", 8)
            self.num_workers = params.get("num_workers", 8)
            
            if mode == "train":
                self.optimizer = params.get("optimizer", "AdamW")
                self.momentum = params.get("momentum", 0.9)
                self.dropout = params.get("dropout", 0.3)
                self.k = params.get("k", 5)
                self.epochs = params.get("epochs", 30)
                self.patience = params.get("patience", 4)
                self.learning_rate = params.get("learning_rate", 0.00001)
                self.weight_decay = params.get("weight_decay", 0.01)
                self.scheduler_lr_patience = params.get("scheduler_lr_patience", 1)
                self.scheduler_lr_factor = params.get("scheduler_lr_factor", 0.2)
                self.train_batch_size = params.get("train_batch_size", 16)
                self.with_class_weights = params.get("with_class_weights", True)
                self.with_transfer_learning = params.get("with_transfer_learning", True)
                self.save_best_model = params.get("save_best_model", True)
    
    class CSVColumns:
        def __init__(self, params):
            """Initialize CSV column names from a dictionary."""
            self.target_col = params.get("target_col", "Group")
            self.patient_col = params.get("patient_col", "Patient number")
            self.seg_col = params.get("seg_col", "Seg path")
            self.slice_idx_col = params.get("slice_idx_col", "Slice idx")
    
    class GlobalParams:
        def __init__(self, params,mode):
            """Initialize global parameters from a dictionary."""
            self.params = params
            self.with_plot = params.get("with_plot", False)
            if mode in ['inference','predict']:
                self.save_preds_per_fold = params.get("save_preds_per_fold",False)
            else:
                self.save_train_val_idx = params.get("save_train_valid_idx", False)
            
            
    
    class SystemPaths:
        def __init__(self, params,mode):
            """Initialize system paths dynamically based on OS and mode."""
            
            self.prepare_paths(mode)  # Get OS-specific paths
            self.dataset_paths = {}
            dataset_type = 'train' if mode == 'train' else 'test'
            self.dataset_paths[dataset_type] = self.get_dataset_paths(params, dataset_type)
            self.dataset_paths[dataset_type]['results_metrices_path'] = os.path.join('results',mode,f"{dataset_type}_metrices_results.csv")
            if mode == 'train':
                self.dataset_paths['oof'] = self.dataset_paths.get('train',None)
                
                
        def get_dataset_paths(self, system_paths, dataset_type):
            """
            Helper function to retrieve dataset paths dynamically.
            Returns (dataset_path, csv_name, csv_path).
            """
            dataset_path = system_paths.get(f"{dataset_type}_dataset_path", "")
            csv_name = system_paths.get(f"{dataset_type}_csv_name", "")
            csv_path = os.path.join(dataset_path, csv_name) if dataset_path and csv_name else ""
            return {'dataset_path':dataset_path, 'csv_name': csv_name, 'csv_path':csv_path}
            
        def prepare_paths(self,mode):
            """Prepare dataset and results paths based on the current OS."""            
            self.results_path = os.path.join('results',mode)
            # Create results directory if it does not exist
            os.makedirs(self.results_path , exist_ok=True)
        
    
    
            
    
                    
        

        


                
                


            
                
                
            
        