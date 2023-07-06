import datetime
import json
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
            "epochs": 50,
            # "batch_size": 60000,
            "validation_split": 0.2,
            "shuffle": True,
            "usercontent" : True
        }
        self.model_name = "model.h5"
        self.sysmane = SysMane.SysMane()
        self.dataset_path = self.sysmane.dataset_path
        self.store_path = self.sysmane.store_path
        self.app_path = self.sysmane.app_path
        self.model = None
        self.model_loaded = False
        self.model_trained = False
        self.use_gpu = True
        self.last_model_acc = 0


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
        if not os.path.exists(self.store_path + "/" + self.model_name):
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
        self.sysmane.write_status("[INFO] Model loaded successfully.",stage="Loading model",percentage=100)
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

        else:
            # Create .prepare_lock file
            self.sysmane.write_status("[INFO] Creating .prepare_lock file.")
            open("{}/{}".format(self.store_path, ".prepare_lock"), "w").close()
        # Check if the dataset is already prepared.
        self.sysmane.write_status("[INFO] Checking dataset...")
        if os.path.exists("dataset"):
            # else:
                # Check if the dataset is complete
                for i in range(10):
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

                    else:
                        self.sysmane.write_status("[OK] Dataset is already prepared. Use force=1 to force prepare.")
                        return "00"
                        #"READY_TO_USE"
                    
                    
                # # Check if the dataset is complete
                # # Scan the training directory for each digit and count the number of files in it.
                # # If the number of files is less than 5000, then the dataset is not complete.
                # # If the dataset is not complete, then force prepare.
                # try:
                #     for i in range(10):
                #         if len(os.listdir("{}/training/{}".format(self.dataset_path, i))) < 5000:
                #             self.sysmane.write_status("[WARN] Data {} is missing. Force prepare.".format(i))
                #             self.sysmane.write_status("Re-initializing (Repair)", 0)
                #             break
                #         if len(os.listdir("{}/validate/{}".format(self.dataset_path, i))) < 1000:
                #             self.sysmane.write_status("[WARN] Data {} is missing. Force prepare.".format(i))
                #             self.sysmane.write_status("Re-initializing (Repair)", 0)
                #             break
                #         else:
                #             self.sysmane.write_status("[OK] Dataset is already prepared. Use force=1 to force prepare.")
                #             return "Dataset is already prepared. Use force=1 to force prepare."
                # except FileNotFoundError:
                #     self.sysmane.write_status("[WARN] Dataset is not prepared. Force prepare.")
                #     self.sysmane.write_status("Re-initializing (Repair)", 0)
                
                # # Check if the dataset is complete
                # # Scan the training directory for each digit and count the number of files in it.
                # # If the number of files is less than 5000, then the dataset is not complete.
                # # If the dataset is not complete, then force prepare.
                # try:
                #     for i in range(10):
                #         if len(os.listdir("{}/training/{}".format(self.dataset_path, i))) < 5000:
                #             self.sysmane.write_status("[WARN] Data {} is missing. Force prepare.".format(i))
                #             self.sysmane.write_status("Re-initializing (Repair)", 0)
                #             break
                #         if len(os.listdir("{}/validate/{}".format(self.dataset_path, i))) < 1000:
                #             self.sysmane.write_status("[WARN] Data {} is missing. Force prepare.".format(i))
                #             self.sysmane.write_status("Re-initializing (Repair)", 0)
                #             break
                #         else:
                #             self.sysmane.write_status("[OK] Dataset is already prepared. Use force=1 to force prepare.")
                #             return "Dataset is already prepared. Use force=1 to force prepare."
                # except FileNotFoundError:
                #     self.sysmane.write_status("[WARN] Dataset is not prepared. Force prepare.")
                #     self.sysmane.write_status("Re-initializing (Repair)", 0)
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

        
    def prepare_dataset(self):       
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
        for i in range(10):
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


        # Prepare dataset finished.
        self.sysmane.write_status("Dataset is prepared.",stage="Preparing dataset",percentage=100)

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
            usercontent_images, usercontent_labels = self.load_mnist("{}/uc".format(self.store_path))
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
        

    def load_mnist(self, path):
        self.sysmane.write_status("Loading MNIST dataset from {}...".format(path), nowrite=True)
        # Load the dataset
        images = []
        labels = []
        for i in range(10):
            progress = int((i+1)/10*100)
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
        # Check if the dataset is prepared.
        if not os.path.exists(self.dataset_path):
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
        if self.use_gpu:
            self.sysmane.write_status("Using GPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            # Showing GPU Name and Memory Usage
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(gpu)
                except RuntimeError as e:
                    print(e)
            
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        else:
            self.sysmane.write_status("Using CPU")
    
        self.sysmane.write_status("Training model", stage="Training model", percentage=0)
        self.sysmane.write_status("Training model at {}".format(self.get_current_time()))
        train_images = train_images.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0 # Normalize the images to a range of [0, 1]
        train_labels = np.eye(10)[train_labels] # One-hot encode the labels
    
        model = tf.keras.models.Sequential()
        model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1))) # Add a convolutional layer with 64 filters, a kernel size of 3x3, and relu activation
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))
    
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
        model = load_model("{}/model/{}".format(self.store_path, self.model_name))
        test_images = test_images.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0
        test_labels = np.eye(10)[test_labels] 

        predictions = model.predict(test_images)
        predictions = np.argmax(predictions, axis=1)

        os.makedirs("result", exist_ok=True)
        for i in range(10):
            os.makedirs("{}/{}".format(self.store_path, i), exist_ok=True)

        for i in range(len(predictions)):
            if predictions[i] != np.argmax(test_labels[i]):
                image = test_images[i].reshape((28, 28))
                image = Image.fromarray(image)
                image.save("{}/{}/{}.png".format(self.store_path, predictions[i], i))

    def test_model_live(self, image):
        image_data = image.read()  # Retrieve the bytes data from the FileStorage object
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        # Convert to grayscale
        image = image.convert("L")
        image = image.resize((28, 28))
        # Convert to numpy array
        image = np.array(image)

        model = load_model("{}/model/{}".format(self.store_path, self.model_name))
        test_images = np.array(image).reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0

        predictions = model.predict(test_images)
        predictions = np.argmax(predictions, axis=1)
        self.sysmane.write_status("Prediction is {} , other predictions are {}".format(predictions[0], predictions), stage="Prediction", percentage=0)
        # Save the image to {storepath}/result/{prediction}--{uuid}.png
        os.makedirs("{}/result".format(self.store_path), exist_ok=True)
        filename = "{}/result/{}--{}.png".format(self.store_path, predictions[0], "uc_" + str(uuid.uuid4()))
        self.save_image(image, filename)
        return predictions
    
    
    def copy_result_to_usercontent(self):
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
            # print("Batch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(batch + 1, logs["loss"], logs["accuracy"]))
            percentage = (batch + 1) / 60000 * 100
            percentage = round(percentage + self.percentage, 3)
            self.sysmane.write_status("Batch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(batch + 1, logs["loss"], logs["accuracy"]), percentage=percentage)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and "loss" in logs and "accuracy" in logs:
            print("Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch + 1, logs["loss"], logs["accuracy"]))
            # Update percentage by Epoch / Total Epochs * 100
            self.percentage = (epoch + 1) / self.params["epochs"] * 100
            percentage = round(self.percentage, 3)
            self.sysmane.write_status("Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch + 1, logs["loss"], logs["accuracy"]), stage="Training model on epoch {}".format(epoch + 1), percentage=percentage)
            # Finish training
            if epoch + 1 == self.params["epochs"]:
                self.sysmane.write_status("Training model finished on epoch {} with accuracy {}".format(epoch + 1, logs["accuracy"]), stage="Training model", percentage=100)
                self.aimane.write_model_acc(logs["accuracy"])