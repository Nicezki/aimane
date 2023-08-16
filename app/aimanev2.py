
import os

try:
    import conmane as cmn
    import sysmane as smn
except:
    from app import conmane as cmn
    from app import sysmane as smn



class AiMane:
    def __init__(self):
        self.sysmane = smn.SysMane()

        self.app_path = self.sysmane.app_path
        self.dataset_path = self.sysmane.dataset_path
        self.store_path = self.sysmane.store_path
        
        # CONFIG LOAD


        # Config will be {self.store_path}/config/config_name.json using join
        self.config_path = os.path.join(self.store_path, "config")
        self.model_config = cmn.ConfigMane("model_config.json", self.config_path)
        self.running_config = cmn.ConfigMane("running_config.json", self.config_path)
        self.prediction_result = cmn.ConfigMane("prediction_result.json", self.config_path)







    if __name__ == "__main__":
        # Print Red Error Message
        print("[ERR] AImane is a library, please run the program from server.py file.")
        exit(0)

        
