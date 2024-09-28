import logging
import time
import datetime

class Logger:
    def __init__(self, filename='log_file.log'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.start_time=time.time()
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # Create a file handler
        self.file_handler = logging.FileHandler(filename)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        
        # Create a console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)
        self.output_console=True
        self.output_file=True

    def log_start(self):
        self.start_time = time.time()
        self.logger.info('Task started at %s',  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def log_end(self):
        end_time = time.time()
        process_time = end_time - self.start_time
        self.logger.info('Task ended at %s', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.logger.info('Task processing time is %s',self.format_process_time( process_time))
        
    def set_console(self,isVisible=True):
        self.output_console=isVisible
        if self.output_console==True:
            self.logger.addHandler(self.console_handler)
        else:
            self.logger.removeHandler(self.console_handler)
        
    def set_file(self,isVisible=True):
        self.output_file=isVisible
        if self.output_file==True:
            self.logger.addHandler(self.file_handler)
        else:
            self.logger.removeHandler(self.file_handler)        

    def log_info(self, message):
        self.logger.info(message)

    def log_debug(self, message):
        self.logger.debug(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)
        
    def format_process_time(self,process_time):
        if process_time < 1:
            return f'{process_time:.4f} seconds'
        elif process_time < 60:
            return f'{process_time:.2f} seconds'
        elif process_time < 3600:
            return f'{process_time / 60:.2f} minutes'
        elif process_time < 86400:
            return f'{process_time / 3600:.2f} hours'
        else:
            return f'{process_time / 86400:.2f} days'
    
    def exit(self):
        self.logger.handlers = [logging.NullHandler()]


