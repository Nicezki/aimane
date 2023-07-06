import datetime
import json
from colorama import Fore
import os


class SysMane:
    
    def __init__(self):
        self.dataset_path = 'dataset'
        self.store_path = 'store'
        self.app_path = 'app'

        self.current_status = {
            'time' : "",
            'status': 'Not initialized',
            'stage': "Starting",
            'percentage': 0
        }

        self.current_log = {
            'time' : "",
            'log': 'Not initialized'
        }
        self.current_env = {
            'time' : "",
            'ver': '0.0.0',
            'logfile' : 0,
            'logoldfile' : 0,
            'statusfile' : 0,
        }
        self.train_status = {
            'time' : "",
            'status': 'Not training',
            'stage': "Not running",
            'percentage': -1,
            'epoch': -1,
            'batch': -1,
            'loss': -1,
            'acc': -1
        }

        self.log_buffer = []
        self.buffer_size = 100  # Number of log entries to buffer before flushing to the file
        self.max_file_size = 1024 * 1024  # Maximum log file size in bytes (1 MB)

        # If .init file not exist in store, first_init will be called
        if not os.path.exists('{}/.init'.format(self.store_path)):
            self.first_init()
        pass


    def setpath(self, dataset_path, store_path, app_path):
        self.dataset_path = dataset_path
        self.store_path = store_path
        self.app_path = app_path

    def getpath(self):
        return self.dataset_path, self.store_path, self.app_path
    
    def getstatus(self):
        return self.current_status
    
    def getlog(self):
        return self.current_log


    def first_init(self):
        # Create necessary directories if they don't exist
        directories = [self.dataset_path, self.store_path, self.app_path]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Create empty files if they don't exist
        files = [
            ('{}/version.txt'.format(self.app_path)),
            # ('{}/status.txt'.format(self.store_path)),
            # ('{}/log.txt'.format(self.store_path)),
            # ('{}/stage.txt'.format(self.store_path))
        ]
        for file_path in files:
            if not os.path.exists(file_path):
                open(file_path, 'a').close()
        
        # Write the current status, stage, version, and log
        current_time = self.get_current_time()
        self.write_status('First time initialization at {}'.format(current_time))
        
        # Create .init file
        open('{}/.init'.format(self.store_path), 'a').close()
        self.write_status('Initialization done at {}'.format(current_time))


    def write_version(self, version):
        # Use app path to store the version
        with open('{}/version.txt'.format(self.app_path), 'w') as f:
            f.write(version)
            print('[{}] : {}'.format(self.get_current_time(), version))
        

    # Function to write the current status
    def write_status(self, status, percentage=None, stage=None, nowrite=False):
        log_message = '[{}]: '.format(self.get_current_time())
        # Use self.current_status to store the status
        self.current_status['time'] = self.get_current_time()
        self.current_status['status'] = status
        if stage is not None:
            self.current_status['stage'] = stage
            log_message += 'Stage {}: '.format(stage)
        if status != '0' and status != 0:
            self.current_status['status'] = status
            log_message += status
        if percentage is not None:
            self.current_status['percentage'] = percentage
            log_message += ' {}%'.format(percentage)

        # Log message rewrite
        # Write the status to log
        if not nowrite:
            self.write_log('{}'.format(status), print_log=True)

        # Send the status back to server.py 


        # Write the status to status.txt
        # DEPRECATED!!!!!!!!!!!!!
        # if not nowrite:
        #     with open('{}/status.txt'.format(self.store_path), 'w') as f:
        #         f.write(json.dumps(self.current_status))


    # def write_log(self, log, print_log=True, raw=False):
    #     # Check if log.txt exist
    #     if os.path.exists('{}/log.txt'.format(self.store_path)):
    #         # Check file size
    #         # If file size is more than 1MB, create a new file, Move the old file to log_old.txt (Append)
    #         if os.path.getsize('{}/log.txt'.format(self.store_path)) > 1000000:
    #             # Check if log_old.txt exist
    #             if os.path.exists('{}/log_old.txt'.format(self.store_path)):
    #                 #Append the log.txt to log_old.txt
    #                 with open('{}/log.txt'.format(self.store_path), 'r') as f:
    #                     log_old = f.read()
    #                 with open('{}/log_old.txt'.format(self.store_path), 'a') as f:
    #                     f.write(log_old)
    #                 # Delete log.txt
    #                 os.remove('{}/log.txt'.format(self.store_path))
    #             # If log_old.txt does not exist, create a new one
    #             else:
    #                 # Rename log.txt to log_old.txt
    #                 os.rename('{}/log.txt'.format(self.store_path), '{}/log_old.txt'.format(self.store_path))
    #         # If file size is less than 1MB, append the log to log.txt
    #         else:
    #             with open('{}/log.txt'.format(self.store_path), 'a') as f:
    #                 f.write('[{}] : {}\n'.format(self.get_current_time(), log))
    #     # If log.txt does not exist, create a new one
    #     else:
    #         with open('{}/log.txt'.format(self.store_path), 'w') as f:
    #             if raw:
    #                 f.write(log)
    #             else:
    #                 f.write('[{}] : {}\n'.format(self.get_current_time(), log))
    #     if print_log:
    #         print('[{}] : {}'.format(self.get_current_time(), log))

    def write_log(self, log, print_log=True, raw=False,force_write=False):
        self.log_buffer.append(log)  # Add log entry to the buffer

        if len(self.log_buffer) >= self.buffer_size or force_write:
            self.flush_buffer()  # Flush buffer to the file if it reaches the buffer size limit
            

        if print_log:
            print('[{}] : {}'.format(self.get_current_time(), log))

    def flush_buffer(self):
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)

        log_file_path = os.path.join(self.store_path, 'log.txt')
        log_old_file_path = os.path.join(self.store_path, 'log_old.txt')

        with open(log_file_path, 'a') as f:
            for log_entry in self.log_buffer:
                f.write('[{}] : {}\n'.format(self.get_current_time(), log_entry))

        self.log_buffer.clear()  # Clear the buffer

        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                logs = f.read()

            with open(log_old_file_path, 'a') as f:
                f.write(logs)

            os.remove(log_file_path)  # Remove the log.txt file


    def get_version(self):
        # Get version from the text file for the first time if not avaliable in self.current_env
        if self.current_env['ver'] == '0.0.0':
        # Handle file not found exception
            try:
                with open('{}/version.txt'.format(self.app_path), 'r') as f:
                    version = f.read()
                    #Return as JSON
                    version = {
                        'time': self.get_current_time(),
                        'version': version
                    }
                    # Convert the version to string
                    version_string = json.dumps(version)
                    
            except FileNotFoundError:
                # Return as JSON 
                version = {
                'time': self.get_current_time(),
                    'version': 'File not found'
                }
                # Convert the version to string
                version_string = json.dumps(version)
            return version_string
        else:
            # Return as JSON
            version = {
                'time': self.get_current_time(),
                'version': self.current_env['ver']
            }
            # Convert the version to string
            version_string = json.dumps(version)
            return version_string
    
    
    def get_status(self,as_json=True):
        #If not as_json, return as dict
        if as_json == False:
            return self.current_status
        # Get status from the self.current_status
        status = self.current_status
        # Convert the status to string
        status_string = json.dumps(status)
        return status_string
    
    

    def get_train_status(self):
        # Get status from the self.current_status
        status = self.current_status
        # Convert the status to string
        status_string = json.dumps(status)
        return status_string
    
    def set_train_status(self, status=None, stage=None, epoch=None, batch=None, loss=None, acc=None):
        # Set time  
        self.train_status['time'] = self.get_current_time()
        # Get status from the self.current_status
        if status is not None:
            self.train_status['status'] = status
        if stage is not None:
            self.train_status['stage'] = stage
        if epoch is not None:
            self.train_status['epoch'] = epoch
        if batch is not None:
            self.train_status['batch'] = batch
        if loss is not None:
            self.train_status['loss'] = loss
        if acc is not None:
            self.train_status['acc'] = acc
        # Convert the status to string
        status_string = json.dumps(self.train_status)
        return status_string
    
    

    def get_current_time(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

    # This is the lib not the main function
    # Prevent the program from running when imported
    
    #Message: Please run the program from the server.py file.
    
    if __name__ == "__main__":
        # Print Red Error Message
        print("{}[ERR] SysMane can't be run as a standalone program." .format(Fore.RED))
        exit(0)


