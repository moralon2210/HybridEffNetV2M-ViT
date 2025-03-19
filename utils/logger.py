import logging
import os
import sys

class ProjectLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProjectLogger, cls).__new__(cls)
            cls._instance._initialized = False
            cls._instance.logger = None
        return cls._instance

    def _initialize_logger(self, mode, log_dir):
        """Initialize or reinitialize the logger with a new directory."""
        if not log_dir:
            raise ValueError("log_dir must be provided (curr_run_folder)")
            
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{mode}.log")

        # ✅ FIX: Remove the logger if it already exists in the root logging system
        if 'ProjectLogger' in logging.Logger.manager.loggerDict:
            old_logger = logging.getLogger('ProjectLogger')
            old_logger.handlers.clear()  # Clear old handlers to prevent duplicates
            logging.Logger.manager.loggerDict.pop('ProjectLogger')  # Remove from root manager

        # ✅ Configure the singleton logger
        self.logger = logging.getLogger('ProjectLogger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # File handler (INFO and ERROR)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8', delay=False)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)

        # Console handler (INFO and ERROR)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        # ✅ Avoid adding duplicate handlers
        if not self.logger.hasHandlers():
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self._initialized = True
        self.logger.info(f"Logger initialized - writing to {log_file}")

    @classmethod
    def get_logger(cls, mode, log_dir=None):
        """Get or create a logger instance."""
        instance = cls()
        if log_dir:
            instance._initialize_logger(mode, log_dir)
        elif not instance._initialized:
            raise ValueError("Logger must be initialized with a log_dir first")
        return instance.logger


def get_logger(mode, log_dir=None):
    """Convenience function to get a logger instance."""
    return ProjectLogger.get_logger(mode, log_dir)

