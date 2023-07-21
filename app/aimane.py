import datetime
import hashlib
import json
import math
import random
import uuid
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import Callback
from app import sysmane as SysMane
from PIL import Image
import base64
import io



class AiMane:
    def __init__(self):
        self.training_config = {
            "use_gpu" : True,
            "epochs": 30,
            "stop_on_acc" : 1.00,
            # "batch_size": 60000,
            "validation_split": 0.2,
            "shuffle": True,
            "usercontent" : True,
            "save_image" : True,
            "save_model" : True,
            "classes" : 10,
            "class_names" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "uc_classes" : 11,
            "uc_class_names" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Star"],
        }
        self.model_name = "model.h5"
        self.sysmane = SysMane.SysMane()
        self.dataset_path = self.sysmane.dataset_path
        self.store_path = self.sysmane.store_path
        self.app_path = self.sysmane.app_path
        self.model = None
        self.model_loaded = False
        self.model_trained = False
        self.last_model_acc = 0
        self.running_config = {
            "train_count" : [],
            "validate_count" : [],
            "last_prediction" : [],
            "last_prediction_uuid" : None,
            "highest_batch" : 0,
        }
        self.prediction_result = {
            "result" : None, 
            "percentage" : None,
            "other_result" : None,
            "other_percentage" : None,
            "uuid" : None,
            "image_data" : None,
            "numpy_array" : None,
            "classes" : None,
            "class_names" : None
        }

        self.sysmane.set_train_status(total_epoch=self.training_config["epochs"])
        # self.training_config["class_names"] combined with self.training_config["classes"] (merged skip duplicate)
        if(self.training_config["usercontent"]):
            self.prediction_result["class_names"] = self.training_config["uc_class_names"]
            self.prediction_result["classes"] = self.training_config["uc_classes"]
        else:
            self.prediction_result["class_names"] = self.training_config["class_names"]
            self.prediction_result["classes"] = self.training_config["classes"]


        # self.training_config["class_names"] = list(dict.fromkeys(self.training_config["class_names"]))
        


    def get_version(self, as_json=True):
        if as_json:
            return self.sysmane.get_version()
        else:
            jsondata = self.sysmane.get_version()
            jsondata = json.loads(jsondata)
            return jsondata["version"]
        

    
    def get_status(self, as_json=True):
        return self.sysmane.get_status(as_json=as_json)
    
    def get_train_status(self):
        return self.sysmane.get_train_status()
    
    def getstatus(self):
        return self.sysmane.getstatus()

    def set_train_status(self, status=None, stage=None, percentage=None, epoch=None, batch=None, loss=None, acc=None, total_epoch=None, finished = None, result=None):
        self.sysmane.set_train_status(status=status, stage=stage, percentage=percentage, epoch=epoch, batch=batch, loss=loss, acc=acc, total_epoch=total_epoch, finished=finished, result=result)


    def set_prediction_result(self, result=None, percentage=None, other_result=None, other_percentage=None, uuid=None, image_data=None,numpy_array=None):
        if result is not None:
            self.prediction_result["result"] = result
        if percentage is not None:
            self.prediction_result["percentage"] = percentage
        if other_result is not None:
            self.prediction_result["other_result"] = other_result
        if other_percentage is not None:
            self.prediction_result["other_percentage"] = other_percentage
        if uuid is not None:
            self.prediction_result["uuid"] = uuid
        if image_data is not None:
            self.prediction_result["image_data"] = image_data
        if numpy_array is not None:
            # Convert ndarray to json
            self.prediction_result["numpy_array"] = numpy_array.tolist()

            

        return self.prediction_result
    
    def set_training_config(self, use_gpu=None, epochs=None, stop_on_acc=None, save_image=None, save_model=None, validation_split=None, shuffle=None, usercontent=None, classes=None, class_names=None, uc_classes=None, uc_class_names=None):
        if use_gpu is not None:
            self.training_config["use_gpu"] = use_gpu
            self.sysmane.write_status("[CONFIG] Training Config: use_gpu is now set to " + str(use_gpu))
        if epochs is not None:
            self.training_config["epochs"] = epochs
            self.sysmane.write_status("[CONFIG] Training Config: epochs is now set to " + str(epochs))
            self.sysmane.set_train_status(total_epoch=epochs)
        if stop_on_acc is not None:
            self.training_config["stop_on_acc"] = stop_on_acc
            self.sysmane.write_status("[CONFIG] Training Config: stop_on_acc is now set to " + str(stop_on_acc))
        if save_image is not None:
            self.training_config["save_image"] = save_image
            self.sysmane.write_status("[CONFIG] Training Config: save_image is now set to " + str(save_image))
        if save_model is not None:
            self.training_config["save_model"] = save_model
            self.sysmane.write_status("[CONFIG] Training Config: save_model is now set to " + str(save_model))
        if validation_split is not None:
            self.training_config["validation_split"] = validation_split
            self.sysmane.write_status("[CONFIG] Training Config: validation_split is now set to " + str(validation_split))
        if shuffle is not None:
            self.training_config["shuffle"] = shuffle
            self.sysmane.write_status("[CONFIG] Training Config: shuffle is now set to " + str(shuffle))
        if usercontent is not None:
            self.training_config["usercontent"] = usercontent
            self.sysmane.write_status("[CONFIG] Training Config: usercontent is now set to " + str(usercontent))
        if classes is not None:
            self.training_config["classes"] = classes
            self.sysmane.write_status("[CONFIG] Training Config: classes is now set to " + str(classes))
        if class_names is not None:
            self.training_config["class_names"] = class_names
            self.sysmane.write_status("[CONFIG] Training Config: class_names is now set to " + str(class_names))
        if uc_classes is not None:
            self.training_config["uc_classes"] = uc_classes
            self.sysmane.write_status("[CONFIG] Training Config: uc_classes is now set to " + str(uc_classes))
        if uc_class_names is not None:
            self.training_config["uc_class_names"] = uc_class_names
            self.sysmane.write_status("[CONFIG] Training Config: uc_class_names is now set to " + str(uc_class_names))


    def get_training_config (self, as_json=True):
        tconf = self.training_config
        if as_json:
            tconf_string = json.dumps(tconf)
            return tconf_string
        else:
            return tconf
        
        
        # self.training_config = {
        #     "use_gpu" : True,
        #     "epochs": 25,
        #     "stop_on_acc" : 1.00,
        #     # "batch_size": 60000,
        #     "validation_split": 0.2,
        #     "shuffle": True,
        #     "usercontent" : True,
        #     "classes" : 10,
        #     "class_names" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        #     "uc_classes" : 11,
        #     "uc_class_names" : ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Star"],
        # }

    


    def get_prediction_result(self,as_json=True):
        status = self.prediction_result
        if as_json:
            status_string = json.dumps(status)
            return status_string
        else:
            return self.prediction_result
        
    
    
    
    def stage_1(self):
        #Check model 
        res_model = self.load_model()

        #If model is not found or cannot be loaded, train a new model
        #Check dataset
        res_dataset = self.check_dataset()
        if res_dataset == "00":
            # Dataset is OK and ready to use, No need to prepare
            self.sysmane.write_status("[INFO] Dataset is OK and ready to use, No need to prepare.")
            return "The dataset is OK and ready to use, No need to prepare."
        if res_dataset == "01":
            # Dataset is missing some file and need to be repaired
            self.sysmane.write_status("[INFO] Dataset is missing some file and need to be repaired.")
            self.prepare_dataset()
            self.sysmane.write_status("Repair completed.")
        if res_dataset == "02":
            # Dataset is being prepared by another process
            # Wait for .prepare_lock to be deleted
            self.sysmane.write_status("[INFO] Dataset is being prepared by another process. Waiting for .prepare_lock to be deleted.")
            while os.path.exists("{}/.prepare_lock".format(self.store_path)):
                pass
        if res_dataset == "03":
            # Dataset is not prepared and need to be prepared (First time)
            self.prepare_dataset()

        if res_dataset == "99":
            # Unknown error
            self.sysmane.write_status("[ERROR] Unknown error while checking dataset.")
            return "There was an unknown error while checking dataset."
        


    def load_model(self):
        self.sysmane.write_status("[INFO] Loading model...",stage="Loading model",percentage=0)
        # Check if model is already loaded
        if self.model_loaded:
            return "01"
            #"MODEL_ALREADY_LOADED"

        # Check if model file exists
        if not os.path.exists("{}/model/{}".format(self.store_path, self.model_name)):
            self.sysmane.write_status("[ERROR] Model file not found.")
            return "02"
            #"MODEL_NOT_FOUND"
        
        # Load model
        self.sysmane.write_status("[INFO] Loading model...",stage="Loading model",percentage=50)
        self.model = load_model("{}/model/{}".format(self.store_path, self.model_name))
        # Check if model is loaded
        if self.model is None:
            self.sysmane.write_status("[ERROR] Model file is corrupted.")
            return "03"
            #"MODEL_CORRUPTED"
        # Set model_loaded to True
        self.sysmane.write_status("[INFO] Model loaded successfully.",stage="Loading model",percentage=100,forcewrite=True)
        self.model_loaded = True
        # Return True
        return "00"
        #"MODEL_LOADED_SUCCESSFULLY"
        
    
    def check_dataset(self):
        self.sysmane.write_status("[INFO] Checking dataset...",stage="Checking dataset",percentage=0)
        #Check dataset for any missing data
        # Check for .prepare_lock file
        # Case of return
        # 00 - Dataset is OK and ready to use, No need to prepare
        # 01 - Dataset is missing some file and need to be repaired
        # 02 - Dataset is being prepared by another process
        # 03 - Dataset is not prepared and need to be prepared (First time)
        # 99 - Unknown error

        if os.path.exists("{}/{}".format(self.store_path, ".prepare_lock")):
            self.sysmane.write_status("[WARN] Dataset is being prepared Please wait.")
            # return "Dataset is being prepared. Please wait. <br> If you want to force prepare, please delete the .prepare_lock file."
            return "02"
            # "PREPARING_IN_PROGRESS"

        # Check if the dataset is already prepared.
        self.sysmane.write_status("[INFO] Checking dataset...")
        if os.path.exists("dataset"):
            # else:
                # Check if the dataset is complete
                for i in range(self.training_config["classes"]):
                    if not os.path.exists("{}/training/{}".format(self.dataset_path, i)):
                        self.sysmane.write_status("[WARN] Data {} is missing. Force prepare.".format(i))
                        self.sysmane.write_status("Re-initializing (Repair) because data {} is missing.".format(i), 0)
                        return "01"
                        #"REPAIR_NEEDED"

                    if not os.path.exists("{}/validate/{}".format(self.dataset_path, i)):
                        self.sysmane.write_status("[WARN] Data {} is missing. Force prepare.".format(i))
                        self.sysmane.write_status("Re-initializing (Repair) because data {} is missing.".format(i), 0)
                        return "01"
                        #"REPAIR_NEEDED"

                    # Count the number of files in training folder then compare with the number of files in validate folder
                    self.get_dataset_count()
                    # Count files in training folder
                    for i in range(self.training_config["classes"]):
                        self.sysmane.write_status("[INFO] Counting files in training folder...",nowrite=True)
                        self.training_count = len(os.listdir("{}/training/{}".format(self.dataset_path, i)))
                        self.sysmane.write_status("[INFO] Counting files in validate folder...",nowrite=True)
                        self.validate_count = len(os.listdir("{}/validate/{}".format(self.dataset_path, i)))
                        # Compare with self.running_config["train_count"] //Array of training count
                        if self.running_config["train_count"][i] == 0:
                            self.sysmane.write_status("[WARN] Metadata count for training is 0. Force prepare.")
                            self.sysmane.write_status("Re-initializing (Repair) because metadata count is 0.", 0)
                            return "01"
                            #"REPAIR_NEEDED"
                        elif self.running_config["validate_count"][i] == 0:
                            self.sysmane.write_status("[WARN] Metadata count for validate is 0. Force prepare.")
                            self.sysmane.write_status("Re-initializing (Repair) because metadata count is 0.", 0)
                            return "01"
                            #"REPAIR_NEEDED"
                        elif self.training_count < self.running_config["train_count"][i]:
                            self.sysmane.write_status("[WARN] Data {} from training is missing. Force prepare.".format(i))
                            self.sysmane.write_status("Re-initializing (Repair) because data {} is missing.".format(i), 0)
                            return "01"
                            #"REPAIR_NEEDED"
                        elif self.training_count > self.running_config["train_count"][i]:
                            # Ignore but warn
                            self.sysmane.write_status("[WARN] Data {} of training is more than expected. but assuming it's OK.".format(i))
                        elif self.validate_count < self.running_config["validate_count"][i]:
                            self.sysmane.write_status("[WARN] Data {} from validate is missing. Force prepare.".format(i))
                            self.sysmane.write_status("Re-initializing (Repair) because data {} is missing.".format(i), 0)
                            return "01"
                            #"REPAIR_NEEDED"
                        elif self.validate_count > self.running_config["validate_count"][i]:
                            # Ignore but warn
                            self.sysmane.write_status("[WARN] Data {} of validate is more than expected. but assuming it's OK.".format(i))
                        else:
                            self.sysmane.write_status("[OK] Data {} is OK.".format(i))
                    

                    else:
                        self.sysmane.write_status("[OK] Dataset is already prepared. Yay~!")
                        return "00"
                        #"READY_TO_USE"

        else:
            self.sysmane.write_status("[WARN] Dataset is not prepared yet. Starting to prepare dataset.")
            return "03"
            #"NOT_PREPARED"
            
        
    def repair_dataset(self):
        # Remove .prepare_lock file
        if os.path.exists("{}/.prepare_lock".format(self.store_path)):
            self.sysmane.write_status("[INFO] Removing .prepare_lock file.")
            os.remove("{}/.prepare_lock".format(self.store_path))
        else:
            self.sysmane.write_status("[INFO] .prepare_lock file not found, Skipping.")
        # Run prepare_dataset
        self.prepare_dataset()


    def count_dataset(self):
        # Count the number of files in training folder then compare with the number of files in validate folder
        # Using count.txt
        self.sysmane.write_status("[INFO] Counting dataset...")
        self.sysmane.write_status("Counting dataset", 0)
        # Count the number of files in training folder
        self.sysmane.write_status("Counting training dataset", 0)
        self.sysmane.write_status("Counting training dataset at {}".format(self.get_current_time()))
        count_training = 0
        for i in range(self.training_config["classes"]):
            count_training += len(os.listdir("{}/training/{}".format(self.dataset_path, i)))
        self.sysmane.write_status("Counting training dataset", 50)
        # Count the number of files in validate folder
        self.sysmane.write_status("Counting validate dataset", 50)
        self.sysmane.write_status("Counting validate dataset at {}".format(self.get_current_time()))
        count_validate = 0
        for i in range(self.training_config["classes"]):
            count_validate += len(os.listdir("{}/validate/{}".format(self.dataset_path, i)))
        self.sysmane.write_status("Counting validate dataset", 100)

    def write_dataset_count(self):
        # Count the number of images in the training and validation sets then save them to a file for later use.
        self.sysmane.write_status("Counting images in training set", percentage=0, nowrite=True)
        for i in range(self.training_config["classes"]):
            # Count the number of images in the training set.
            self.sysmane.write_status("Counting images in training set for class {}".format(i), percentage=int((i+1)/self.training_config["classes"]*100), nowrite=True)
            train_count = len(os.listdir("{}/training/{}".format(self.dataset_path, i)))
            self.sysmane.write_status("Counting images in training set for class {} finished. {} images found.".format(i, train_count), percentage=int((i+1)/self.training_config["classes"]*100), nowrite=True)
            # Count the number of images in the validation set.
            self.sysmane.write_status("Counting images in validation set for class {}".format(i), percentage=int((i+1)/self.training_config["classes"]*100), nowrite=True)
            validate_count = len(os.listdir("{}/validate/{}".format(self.dataset_path, i)))
            self.sysmane.write_status("Counting images in validation set for class {} finished. {} images found.".format(i, validate_count), percentage=int((i+1)/self.training_config["classes"]*100), nowrite=True)
            # Save to array, Will write to file later.
            self.running_config["train_count"].append(train_count)
            self.running_config["validate_count"].append(validate_count)
        self.sysmane.write_status("Writing training and validation counts to file", percentage=0, nowrite=True)
        with open("{}/count.txt".format(self.dataset_path), "w") as f:
            for i in range(self.training_config["classes"]):
                self.sysmane.write_status("Writing training and validation counts to file for class {}".format(i), percentage=int((i+1)/self.training_config["classes"]*100), nowrite=True)
                f.write("{}\n".format(self.running_config["train_count"][i]))
                f.write("{}\n".format(self.running_config["validate_count"][i]))
        self.sysmane.write_status("Writing training and validation counts to file finished.", percentage=100, nowrite=True)

    def get_dataset_count(self):
        # From count.txt
        self.sysmane.write_status("Getting training and validation counts from file", percentage=0, nowrite=True)
        # Check if count.txt exists
        if not os.path.exists("{}/count.txt".format(self.dataset_path)):
            self.sysmane.write_status("count.txt not found. Creating count.txt", percentage=0, nowrite=True)
            self.write_dataset_count()

        with open("{}/count.txt".format(self.dataset_path), "r") as f:
            # Clear the array first
            self.running_config["train_count"] = []
            self.running_config["validate_count"] = []
            for i in range(self.training_config["classes"]):
                self.sysmane.write_status("Getting training and validation counts from file for class {}".format(i), percentage=int((i+1)/self.training_config["classes"]*100), nowrite=True)
                # Line 1 to self.training_config["classes"] is training count
                # Line self.training_config["classes"]+1 to self.training_config["classes"]*2 is validation count
                # Preventing read empty line
                train_count = f.readline()
                if train_count == "":
                    train_count = f.readline()
                self.running_config["train_count"].append(int(train_count))
                validate_count = f.readline()
                if validate_count == "":
                    validate_count = f.readline()
                self.running_config["validate_count"].append(int(validate_count))

                
                
        self.sysmane.write_status("Getting training and validation counts from file finished.", percentage=100)
        self.sysmane.write_status("Training count: {}".format(self.running_config["train_count"]))
        self.sysmane.write_status("Validate count: {}".format(self.running_config["validate_count"]))
        self.sysmane.write_status("Total training count: {}".format(sum(self.running_config["train_count"])))
        self.sysmane.write_status("Total validate count: {}".format(sum(self.running_config["validate_count"])), forcewrite=True)


   
    def prepare_dataset(self):     
        # Create .prepare_lock file
        if not os.path.exists("{}/{}".format(self.store_path, ".prepare_lock")):
            self.sysmane.write_status("[INFO] Creating .prepare_lock file.")
            open("{}/{}".format(self.store_path, ".prepare_lock"), "w").close()  


        # Prepare the dataset
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        self.sysmane.write_status("Dataset is being prepared. This may take a while.",stage="Preparing dataset",percentage=0)
        self.sysmane.write_status("Loading dataset at {}".format(self.get_current_time()))
        self.sysmane.write_status("Loading dataset", 0)

        # Create the dataset directory if it doesn't exist.
        if not os.path.exists("{}/dataset".format(self.dataset_path)):
            self.sysmane.write_status("Creating dataset directory")
            self.sysmane.write_status("Dataset directory is missing. Creating dataset directory.")
            os.mkdir("{}/dataset".format(self.dataset_path))
            # Check if the dataset directory is created.
            if os.path.exists("{}/dataset".format(self.dataset_path)):
                self.sysmane.write_status("[OK] Dataset directory is created.")
            else:
                self.sysmane.write_status("[ERROR] Dataset directory is not created.")
                return "Dataset directory is not created."
        else:
            self.sysmane.write_status("[SKIP] Dataset directory already exists.")
            

        # Split the training data into training and validation sets.
        self.sysmane.write_status("Splitting dataset", 0)
        split_index = int(train_images.shape[0] * self.training_config["validation_split"])
        self.sysmane.write_status("Splitting dataset at {}".format(self.get_current_time()))
        self.sysmane.write_status("Split index: {}".format(split_index))
        train_images, validate_images = train_images[:split_index], train_images[split_index:]
        self.sysmane.write_status("Train images: {}".format(train_images.shape))
        train_labels, validate_labels = train_labels[:split_index], train_labels[split_index:]
        self.sysmane.write_status("Train labels: {}".format(train_labels.shape))

        # Save the training and validation sets to separate directories.
        for i in range(self.training_config["classes"]):
            progress = int((i+1)/10*100)
            self.sysmane.write_status("Creating directory {}/training/{}".format(self.dataset_path, i), progress)
            os.makedirs("{}/training/{}".format(self.dataset_path, i), exist_ok=True)

            progress = int((i+1)/10*100)
            self.sysmane.write_status("Creating directory {}/validate/{}".format(self.dataset_path, i), progress)
            os.makedirs("{}/validate/{}".format(self.dataset_path, i), exist_ok=True)

        for idx, (image, label) in enumerate(zip(train_images, train_labels)):
            progress = int((idx+1)/train_images.shape[0]*100)
            #Skip if the image is already saved.
            if os.path.exists("{}/training/{}/{}.png".format(self.dataset_path, label, idx)):
                self.sysmane.write_status("Skipping image {} to {} because it is already saved.".format(idx, "{}/training/{}/{}.png".format(self.dataset_path, label, idx)),nowrite=True,percentage=progress)
                continue
            filename = "{}/training/{}/{}.png".format(self.dataset_path, label, idx)
            self.sysmane.write_status("Saving image {} to {}".format(idx, filename),nowrite=True)
            self.sysmane.write_status("Saving  train image {} to {}".format(idx, filename), progress)
            self.save_image(image, filename)

        for idx, (image, label) in enumerate(zip(validate_images, validate_labels)):
            #Skip if the image is already saved.
            if os.path.exists("{}/validate/{}/{}.png".format(self.dataset_path, label, idx)):
                self.sysmane.write_status("Skipping image {} to {} because it is already saved.".format(idx, "{}/validate/{}/{}.png".format(self.dataset_path, label, idx)),nowrite=True)
                continue
            progress = int((idx+1)/validate_images.shape[0]*100)
            filename = "{}/validate/{}/{}.png".format(self.dataset_path, label, idx)
            self.sysmane.write_status("Saving image {} to {}".format(idx, filename),nowrite=True)
            self.sysmane.write_status("Saving validate image {} to {}".format(idx, filename), progress)
            self.save_image(image, filename)

        # Write the number of images in the training and validation sets to a file.
        self.write_dataset_count()

        # Delete .prepare_dataset file
        if os.path.exists("{}/.prepare_dataset".format(self.store_path)):
            os.remove("{}/.prepare_dataset".format(self.store_path))

        # Prepare dataset finished.
        self.sysmane.write_status("Dataset is prepared.",stage="Preparing dataset",percentage=100,forcewrite=True)



    def save_image(self ,image, filename):
        self.sysmane.write_status("Converting image {} to float32...".format(filename),nowrite=True)
        # Progrss from filename
        self.sysmane.write_status("Normalizing image {}...".format(filename),nowrite=True)
        image = (image / 255.0).astype(np.float32)
        self.sysmane.write_status("Converting image {} to float32...".format(filename),nowrite=True)
        self.sysmane.write_status("Scaling image {}...".format(filename),nowrite=True)
        # Scale the pixel values to the range [0, 255] and convert to uint8
        image = (image * 255).astype(np.uint8)
        self.sysmane.write_status("Saving image {}...".format(filename),nowrite=True)
        # Save the image
        cv2.imwrite(filename, image)
        # Check if the image is saved.
        if os.path.exists(filename):
            self.sysmane.write_status("[OK] Image {} is saved.".format(filename),nowrite=True)
        else:
            self.sysmane.write_status("[ERROR] Image {} is not saved.".format(filename))


    def load_dataset(self):
        self.sysmane.write_status("Loading dataset...", stage="Loading dataset", percentage=0)
        # Load the dataset
        train_images, train_labels = self.load_mnist("{}/training".format(self.dataset_path))
        validate_images, validate_labels = self.load_mnist("{}/validate".format(self.dataset_path))
        # If usercontent is True, also load the usercontent dataset.
        if self.training_config["usercontent"]:
            #Check if the usercontent dataset exists.
            if not os.path.exists("{}/uc".format(self.store_path)):
                self.sysmane.write_status("[ERROR] Usercontent dataset does not exist.")
                return "Usercontent dataset does not exist."
            # Load the usercontent dataset.
            usercontent_images, usercontent_labels = self.load_mnist("{}/uc".format(self.store_path), usercontent=True)
            if usercontent_images is not None and usercontent_labels is not None and train_images is not None and train_labels is not None:
                # Concatenate the usercontent dataset to the training dataset.
                train_images = np.concatenate((train_images, usercontent_images))
                train_labels = np.concatenate((train_labels, usercontent_labels))
                # Count the number of usercontent images.
                usercontent_count = usercontent_images.shape[0]
                self.sysmane.write_status("Usercontent dataset is loaded. {} images are loaded.".format(usercontent_count))
            else:
                self.sysmane.write_status("[WARNING!] Usercontent dataset is not loaded or training dataset is not loaded.")
        else:
            self.sysmane.write_status("[WARNING!] Usercontent dataset is not loaded.")

        # Check if the dataset is loaded.
        if train_images is None or train_labels is None or validate_images is None or validate_labels is None:
            self.sysmane.write_status("[ERROR] Dataset is not loaded.")
            return "Dataset is not loaded."
        else:
            self.sysmane.write_status("[OK] Dataset is loaded.")
            return train_images, train_labels, validate_images, validate_labels
        

    def load_mnist(self, path, usercontent=False):
        self.sysmane.write_status("Loading MNIST dataset from {}...".format(path), nowrite=True)
        # Load the dataset
        images = []
        labels = []
        if usercontent:
            training_classes = self.training_config["uc_classes"]
        else:
            training_classes = self.training_config["classes"]

        for i in range(training_classes):
            progress = int((i+1)/self.training_config["classes"]*100)
            #Check if the directory exists.
            if not os.path.exists("{}/{}".format(path, i)):
                self.sysmane.write_status("[ERROR] Directory {}/{} does not exist.".format(path, i))
                # Skip if the directory does not exist.
                continue
            self.sysmane.write_status("Loading directory {}/{}".format(path, i), progress)
            for filename in os.listdir("{}/{}".format(path, i)):
                progress = int((i+1)/10*100)
                self.sysmane.write_status("Loading image {}/{}".format(i, filename), progress, nowrite=True)
                image = cv2.imread("{}/{}/{}".format(path, i, filename), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    self.sysmane.write_status("[ERROR] Image {}/{} is not loaded.".format(i, filename))
                    return None, None
                else:
                    images.append(image)
                    labels.append(i)
        images = np.array(images)
        labels = np.array(labels)
        self.sysmane.write_status("Loading MNIST dataset from {} finished.".format(path))
        return images, labels
    
    def train(self):
        self.sysmane.write_status("Training model starting...", stage="Training model", percentage=0)
        # Check if the dataset is prepared and have dataset in it. check if count.txt exists inside the dataset pa
        if not os.path.exists(self.dataset_path or not os.path.exists("{}/dataset".format(self.dataset_path or not os.path.exists("{}/count.txt".format(self.dataset_path))))):
            self.sysmane.write_status("[ERROR] Dataset is not prepared.")
            return "Dataset is not prepared."
        else:
            self.sysmane.write_status("[OK] Dataset is prepared.")


        # Load the dataset
        self.sysmane.write_status("Loading dataset...", stage="Training model", percentage=0)
        train_images, train_labels, validate_images, validate_labels = self.load_dataset()
        # Check if the dataset is loaded.
        if train_images is None or train_labels is None or validate_images is None or validate_labels is None:
            self.sysmane.write_status("[ERROR] Dataset is not loaded.")
            return "Dataset is not loaded." 
        else:
            self.sysmane.write_status("[OK] Dataset is loaded.")

        # Train the model
        self.train_model(train_images, train_labels)


    def check_model_before_training(self):
        # Check if the model is trained.
        if os.path.exists("{}/model.h5".format(self.store_path)):
            self.sysmane.write_status("[WARNING] Model is already trained.")
            self.last_model_acc = self.get_model_acc()
            self.sysmane.write_status("Last model accuracy: {}".format(self.last_model_acc))
            self.sysmane.write_status("If new model accuracy is lower than last model accuracy, the model will be discarded.")
            # Rename the model file to model_trained
            os.rename("{}/model.h5".format(self.store_path), "{}/model_{}.h5".format(self.store_path, self.last_model_acc))
            return False
        else:
            self.sysmane.write_status("[OK] Model is not trained.")
            return True


    def train_model(self, train_images, train_labels):
        # USE GPU or CPU
        if self.training_config["use_gpu"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            config = tf.compat.v1.ConfigProto()
            session = tf.compat.v1.Session(config=config)
            self.sysmane.write_status("GPU options are enabled. Try to use GPU.")
            # Prevent alreay allocated memory from being allocated again.
            try: 
                tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
            except:
                self.sysmane.write_status("[WARNING] GPU allocation is failed. Maybe It's already allocated.")
            # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

        self.sysmane.write_status("Training model", stage="Training model", percentage=0)
        self.sysmane.write_status("Training model at {}".format(self.get_current_time()))
        train_images = train_images.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0 # Normalize the images to a range of [0, 1]
        classes = self.training_config["uc_classes"] 
        train_labels = np.eye(classes)[train_labels]
        model = tf.keras.models.Sequential()
        model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1))) # Add a convolutional layer with 64 filters, a kernel size of 3x3, and relu activation
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(classes, activation="softmax"))
    
        model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
        # Create an instance of the VerboseCallback class, passing the `sysname` object
        verbose_callback = VerboseCallback(self)
        # Create and configure the verbose callback
        callbacks = [verbose_callback]
        # callbacks[0].sysmane = self.sysmane  # Set a reference to the sysmane object
    
        # Start training
        model.fit(train_images, train_labels, epochs=self.training_config["epochs"], callbacks=callbacks)
    
        os.makedirs("{}/model".format(self.store_path), exist_ok=True)
        current_model_acc = self.get_model_acc()
        self.sysmane.write_status("Current model accuracy: {}".format(current_model_acc))
        # Give back VRAM to the system
        if self.training_config["use_gpu"]:
            tf.keras.backend.clear_session()
            self.sysmane.write_status("GPU options are enabled. Try to give back VRAM to the system.")


        # Check if the current model accuracy is lower than last model accuracy
        if isinstance(self.last_model_acc, int) and isinstance(current_model_acc, int) and self.last_model_acc > current_model_acc:
            self.sysmane.write_status("Last model accuracy is higher than current model accuracy. Discarding current model.")
            # Rename the model file to model_discarded_{accuracy}.h5
            model.save("{}/model/model_discarded_{}.h5".format(self.store_path, current_model_acc))
            #Rename back the last model file to model.h5
            os.rename("{}/model/model_{}.h5".format(self.store_path, self.last_model_acc), "{}/model/model.h5".format(self.store_path))
        else:
            self.sysmane.write_status("Current model accuracy is higher than last model accuracy. Saving current model.")
            # Save the model
            model.save("{}/model/model.h5".format(self.store_path))
            self.sysmane.write_status("Model saved at path: {}/model".format(self.store_path))
            self.sysmane.write_status("Training model finished with accuracy: {}".format(current_model_acc))
        
        # self.sysmane.write_status("Model saved at path: {}/model/{}".format(self.store_path, self.model_name))
        # self.sysmane.write_status("Training model finished", stage="Training model", percentage=100)
    

    def test_model(self, test_images, test_labels):
        self.sysmane.write_status("Testing model", stage="Testing model", percentage=0)
        load_result = self.load_model()
        #00 - Model loaded successfully
        #01 - Model already loaded
        #02 - Model file not found
        #03 - Model file is corrupted

        if load_result == "02":
            self.sysmane.write_status("Model file not found. Please train the model first.")
            return False
        elif load_result == "03":
            self.sysmane.write_status("Model file is corrupted. Please train the model again.")
            return False
        
        model = self.model

        # Check if the model is loaded successfully
        if model == None:
            self.sysmane.write_status("Model is not loaded. Please train the model first.")
            return False

        test_images = test_images.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0
        test_labels = np.eye(10)[test_labels] 

        predictions = model.predict(test_images)
        predictions = np.argmax(predictions, axis=1)

        os.makedirs("result", exist_ok=True)
        for i in range(self.training_config["classes"]):
            os.makedirs("{}/{}".format(self.store_path, i), exist_ok=True)

        for i in range(len(predictions)):
            if predictions[i] != np.argmax(test_labels[i]):
                image = test_images[i].reshape((28, 28))
                image = Image.fromarray(image)
                image.save("{}/{}/{}.png".format(self.store_path, predictions[i], i))

    def test_model_live(self, image):
        load_result = self.load_model()
        #00 - Model loaded successfully
        #01 - Model already loaded
        #02 - Model file not found
        #03 - Model file is corrupted

        if load_result == "02":
            self.sysmane.write_status("Model file not found. Please train the model first.")
            return "Model file not found. Please train the model first."
        
        elif load_result == "03":
            self.sysmane.write_status("Model file is corrupted. Please train the model again.")
            return "Model file is corrupted. Please train the model again."
        
        
        model = self.model

        # Check if the model is loaded successfully
        if model == None:
            self.sysmane.write_status("Model is not loaded. Please train the model first.")
            return "Model is not loaded. Please train the model first."
        


        image_data = image.read()  # Retrieve the bytes data from the FileStorage object
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        # Convert to grayscale
        image = image.convert("L")
        image = image.resize((28, 28))
        # Convert to numpy array
        image = np.array(image)

        test_images = np.array(image).reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0

        predictions = model.predict(test_images)
        predictions = np.argmax(predictions, axis=1)
        # percentage = np.max(predictions[0])
        # prediction_list = [predictions[0]]
        other_result = []
        self.sysmane.write_status("Prediction is {} , other predictions are {}".format(predictions[0], predictions), stage="Prediction", percentage=0)

        if not self.training_config["save_image"]:
            self.sysmane.write_status("Image saving is disabled. Skipping image saving")
            return predictions
        # Save the image to {storepath}/result/{prediction}--{uuid}.png
        os.makedirs("{}/result".format(self.store_path), exist_ok=True)
        # File name will determine by the image data to avoid duplicate dataset
        filename = hashlib.md5(image_data).hexdigest()
        # Banned filename will be discarded
        # Blank image is 61dce78ef88be2d672e96f83c403fd73
        if filename == "61dce78ef88be2d672e96f83c403fd73":
            self.sysmane.write_status("Blank Image Detected!! Discarding blank image")
            return predictions
        

        filepath = "{}/result/{}--{}.png".format(self.store_path, predictions[0], "uc_" + filename)
        self.save_image(image, filepath)
        self.running_config["last_prediction"] = predictions[0]
        self.running_config["last_prediction_uuid"] = "uc_" + filename
        # Image data  will be provided to the frontend by api to show the image
        image_data = base64.b64encode(image_data).decode("utf-8")
        #self.set_prediction_result(result=int(predictions[0].argmax()), percentage=float(percentage), other_result=other_result, other_percentage=other_percentage, uuid="uc_" + filename, image_data=image_data)
        self.set_prediction_result(result=int(predictions[0]), other_result=other_result, uuid="uc_" + filename, image_data=image_data, numpy_array=test_images)
        return predictions
    

    def define_prediction(self,new_prediction, prediction, uuid):
        # Rename file in /result/{prediction}--{uuid}.png to /result/{new_prediction}--{uuid}.png
        # Check if the file exist
        if os.path.isfile("{}/result/{}--{}.png".format(self.store_path, prediction, uuid)):
            # Check if rename name is already exist or not
            if os.path.isfile("{}/result/{}--{}.png".format(self.store_path, new_prediction, uuid)):
                self.sysmane.write_status("Prediction already exist", stage="Prediction", percentage=0)
                # Remove the file
                os.remove("{}/result/{}--{}.png".format(self.store_path, prediction, uuid))
                return False
            os.rename("{}/result/{}--{}.png".format(self.store_path, prediction, uuid), "{}/result/{}--{}.png".format(self.store_path, new_prediction, uuid))
            self.sysmane.write_status("Prediction defined to {}".format(new_prediction), stage="Prediction", percentage=0)
            return True
        else:
            self.sysmane.write_status("Prediction not found", stage="Prediction", percentage=0)
            return False
        
    def define_last_prediction(self, new_prediction):
        if not self.training_config["save_image"]:
            self.sysmane.write_status("Image saving is disabled. Cannot define prediction")
            return False
        # Get last prediction and uuid from running_config
        prediction = self.running_config["last_prediction"]
        uuid = self.running_config["last_prediction_uuid"]
        return self.define_prediction(new_prediction, prediction, uuid)
    

    def rewrite_filename(self):
        self.sysmane.write_status("Rewriting filename", stage="Rewriting filename", percentage=0)
        # Use this for upgrade from uuid to hashlib md5
        # Copy back all images in {storepath}/uc/{label} to {storepath}/result
        #{storepath}/uc/{label}/{uuid}.png => {storepath}/result/{prediction}--{uuid}.png
        for label in os.listdir("{}/uc".format(self.store_path)):
            for filename in os.listdir("{}/uc/{}".format(self.store_path, label)):
                uuid = filename.split(".")[0]
                prediction = label
                shutil.copyfile("{}/uc/{}/{}".format(self.store_path, label, filename), "{}/result/{}--{}.png".format(self.store_path, prediction, uuid))
        # Delete all images in {storepath}/uc
        shutil.rmtree("{}/uc".format(self.store_path))
        os.makedirs("{}/uc".format(self.store_path), exist_ok=True)

        # 1. Loop all images in {storepath}/result
        # 2. Read the image
        # 3. Get the Hashlib md5 of the image
        # 4. Copy the image to {storepath}/uc/{label}/{uuid}.png

        for filename in os.listdir("{}/result".format(self.store_path)):
            self.sysmane.write_status("Reading {}".format(filename))
            image_data = open("{}/result/{}".format(self.store_path, filename), "rb").read()
            self.sysmane.write_status("Hashing {}".format(filename))
            hmd5 = hashlib.md5(image_data).hexdigest()
            self.sysmane.write_status("Covert {} to {}".format(filename, hmd5))
            prediction = filename.split("--")[0]
            self.sysmane.write_status("Copying {} to {}/uc/{}/{}.png".format(filename, self.store_path, prediction, hmd5))
            os.makedirs("{}/uc/{}".format(self.store_path, prediction), exist_ok=True)
            shutil.copyfile("{}/result/{}".format(self.store_path, filename), "{}/uc/{}/{}.png".format(self.store_path, prediction, hmd5))

        # Delete all images in {storepath}/result
        self.sysmane.write_status("Deleting all images in {}/result".format(self.store_path))
        shutil.rmtree("{}/result".format(self.store_path))
        self.sysmane.write_status("Creating {}/result".format(self.store_path))
        os.makedirs("{}/result".format(self.store_path), exist_ok=True)
        self.sysmane.write_status("Copying result to user content completed",forcewrite=True)

        
    
    def copy_result_to_usercontent(self):
        self.sysmane.write_status("Copying result to user content", stage="Copying result to user content", percentage=0)
        # Use this for upgrade from uuid to hashlib md5
        # Copy all images in {storepath}/result to {storepath}/uc/{label}
            #{storepath}/result/{prediction}--{uuid}.png => {storepath}/uc/{label}/{uuid}.png
        for filename in os.listdir("{}/result".format(self.store_path)):
            prediction = filename.split("--")[0]
            uuid = filename.split("--")[1].split(".")[0]
            os.makedirs("{}/uc/{}".format(self.store_path, prediction), exist_ok=True)
            shutil.copyfile("{}/result/{}".format(self.store_path, filename), "{}/uc/{}/{}.png".format(self.store_path, prediction, uuid))
        # Delete all images in {storepath}/result
        shutil.rmtree("{}/result".format(self.store_path))
        os.makedirs("{}/result".format(self.store_path), exist_ok=True)
        self.sysmane.write_status("Copying result to user content completed", stage="Copying result to user content", percentage=100)

            
    # def test_model_label(self, image, label):   
    

    # Add Images to Training dataset
    def add_train_dataset(self, image, label):
        # Save the image to usercontent folder
        os.makedirs("{}/{}".format(self.store_path, label), exist_ok=True)
        filename = "{}/{}/{}.png".format(self.store_path, label, "uc_" + str(uuid.uuid4()))
        self.save_image(image, filename)
        self.sysmane.write_status("User content saved at {}".format(filename))

    

    def get_current_time(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def write_model_acc(self, acc):
        # Check if {storepath}/model/acc.txt exists
        if not os.path.exists("{}/model/acc.txt".format(self.store_path)):
            # Create {storepath}/model/acc.txt
            os.makedirs("{}/model".format(self.store_path), exist_ok=True)
            open("{}/model/acc.txt".format(self.store_path), "w").close()
            self.sysmane.write_status("Created {}/model/acc.txt".format(self.store_path))
        # Replace the acc.txt file with new acc
        filename = "{}/model/acc.txt".format(self.store_path)
        with open(filename, "w") as f:
            f.write("{}".format(acc))
        self.sysmane.write_status("Model accuracy saved at {}".format(filename))

        # Write model acc history to acc_history.txt
        with open("{}/model/history.txt".format(self.store_path), "a") as f:
            f.write("{}\n".format("Model accuracy: {}, Trained at: {}".format(acc, self.get_current_time())))


    def get_model_acc(self):
        # Check if {storepath}/model/acc.txt exists
        if not os.path.exists("{}/model/acc.txt".format(self.store_path)):
            return 0
        
        # Read from {storepath}/model/acc.txt
        with open("{}/model/acc.txt".format(self.store_path), "r") as f:
            return f.read()
        
    # This is the lib not the main function
    # Prevent the program from running when imported
    
    #Message: Please run the program from the server.py file.
    
    if __name__ == "__main__":
        print("[ERR] :( Please run the program from the server.py file!!!!!!")
        exit(0)

class VerboseCallback(Callback):
    def __init__(self, aimane):
        super(VerboseCallback, self).__init__()
        self.sysmane = aimane.sysmane
        self.aimane = aimane
        self.percentage = 0


    def on_batch_end(self, batch, logs=None):
        if logs is not None and "loss" in logs and "accuracy" in logs:
            # #Save highest batch for calculating percentage in self.aimane.self.running_config["highest_batch"]
            # if batch > self.aimane.running_config["highest_batch"]:
            #     self.aimane.running_config["highest_batch"] = batch
            
            # # print("Batch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(batch + 1, logs["loss"], logs["accuracy"]))
            # if self.aimane.running_config["highest_batch"] == 0 or self.percentage == 0:
            #     percentage = (batch + 1) / 60000 * 100 
            # else:
            #     # Percentage of 1 epoch
            #     percentage = ((batch + 1) / self.aimane.running_config["highest_batch"] * 100) 
            #     # Weight of 1 epoch to the total percentage
            #     percentage = percentage / 100 * (1 / self.aimane.training_config["epochs"])
            percentage = percentage = (batch + 1) / 60000 * 100 
            # Cut the zeros after 3 decimal places
            percentage = "{:.3f}".format(percentage + self.percentage)
            self.sysmane.write_status("Batch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(batch + 1, logs["loss"], logs["accuracy"]), percentage=percentage)
            # self.sysmane.set_train_status(status=status, stage=stage, percentage=percentage, epoch=epoch, batch=batch, loss=loss, acc=acc)
            self.sysmane.set_train_status(status="Training model", stage="Training model on batch {}".format(batch + 1), percentage=percentage, epoch=None, batch=batch, loss=logs["loss"], acc=logs["accuracy"], finished=False, result="Not ready")

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and "loss" in logs and "accuracy" in logs:
            # If epoch is 0 clear history
            if epoch == 0:
                self.sysmane.clear_train_history()
            print("Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch + 1, logs["loss"], logs["accuracy"]))
            # Update percentage by Epoch / Total Epochs * 100
            self.percentage = (epoch + 1) / self.params["epochs"] * 100
            percentage = "{:.3f}".format(self.percentage)
            self.sysmane.write_status("Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch + 1, logs["loss"], logs["accuracy"]), stage="Training model on epoch {}".format(epoch + 1), percentage=percentage)
            self.sysmane.set_train_status(status="Training model", stage="Training model on epoch {}".format(epoch + 1), percentage=percentage, epoch=epoch+1, batch=None, loss=logs["loss"], acc=logs["accuracy"])
            if self.aimane.training_config["stop_on_acc"] is not None and logs["accuracy"] >= self.aimane.training_config["stop_on_acc"] and self.model is not None:
                self.model.stop_training = True
                self.sysmane.write_status("Training model finished early on epoch {} with satisfied accuracy {}".format(epoch + 1, logs["accuracy"]), stage="Training model", percentage=100,forcewrite=True)
                self.aimane.write_model_acc(logs["accuracy"])
                self.sysmane.set_train_status(status="Finished training", stage="Finished early on epoch {}".format(epoch + 1), finished=True, result="Training model finished early on epoch {} with satisfied accuracy {}".format(epoch + 1, logs["accuracy"]), percentage=100)
            elif self.params is not None and epoch + 1 == self.params["epochs"]:
                self.sysmane.write_status("Training model finished on epoch {} with accuracy {}".format(epoch + 1, logs["accuracy"]), stage="Training model", percentage=100,forcewrite=True)
                self.aimane.write_model_acc(logs["accuracy"])
                self.sysmane.set_train_status(status="Finished training", stage="Finished on epoch {}".format(epoch + 1), finished=True, result="Training model finished on set epoch {} with accuracy {}".format(epoch + 1, logs["accuracy"]), percentage=100)

            #Check if training is finished (Spare case if model.stop_training is not working)
            if self.model is not None and self.model.stop_training:
                self.sysmane.write_status("Training model finished early on epoch {} with accuracy {}".format(epoch + 1, logs["accuracy"]), stage="Training model", percentage=100,forcewrite=True)
                self.aimane.write_model_acc(logs["accuracy"])
                self.sysmane.set_train_status(status="Finished training", stage="Finished early on epoch {}".format(epoch + 1), finished=True, result="Training model finished early on epoch {} with accuracy {}".format(epoch + 1, logs["accuracy"]), percentage=100)




    # This is the lib not the main function
    # Prevent the program from running when imported
    if __name__ == "__main__":
        # Print Red Error Message
        print("[ERR] AImane is a library, please run the program from server.py file.")
        exit(0)

