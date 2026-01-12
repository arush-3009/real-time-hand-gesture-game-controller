from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter
import pandas as pd

PATH_TO_RAW_CLASS_IMAGES = "dataset/raw"

class DataSplitter:

    def __init__(self, raw_img_dir_path):
        self.raw_img_dir = Path(raw_img_dir_path)
    
    def get_raw_data(self):
        self.X_gesture_images = []
        self.y_gesture_classes = []
        self.gestures = []
        gesture_directories = list(self.raw_img_dir.iterdir())
        for gesture_dir in gesture_directories:
            gesture_class = gesture_dir.name
            if gesture_class not in self.gestures:
                self.gestures.append(gesture_class)
            print(f"\nNow processing -> Gesture: {gesture_class}")
            gesture_images = list(gesture_dir.glob("*.jpg"))
            for img in gesture_images:
                self.X_gesture_images.append(img)
                self.y_gesture_classes.append(gesture_class)
        
        self.gestures = sorted(self.gestures)
        print(f"\nAll images now in 1 list and corresponding class labels in another list ->")
        print(f"\nList containing all image paths -> X_gesture_images\nSize: {len(self.X_gesture_images)} image paths.")
        print(f"\nList containing all class labels -> y_gesture_classes\nSize: {len(self.y_gesture_classes)} labels.")

    def split_data(self):
        """
        Returns True if succesfully split, False otherwise.
        """
        if len(self.X_gesture_images) == 0 or len(self.y_gesture_classes) == 0:
            return False
        
        X_temp, X_test, y_temp, y_test = train_test_split(self.X_gesture_images, self.y_gesture_classes, test_size=0.15, random_state=78, stratify=self.y_gesture_classes)

        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, stratify=y_temp, random_state=78)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        train_counts = Counter(self.y_train)
        val_counts = Counter(self.y_val)
        test_counts = Counter(self.y_test)

        stats = {"Train": [], "Train %": [], "Validation": [], "Validation %": [], "Test": [], "Test %": []}

        for gesture in self.gestures:
            stats["Train"].append(train_counts[gesture])
            stats["Train %"].append(100 * (train_counts[gesture] / len(self.y_train)))
            stats["Validation"].append(val_counts[gesture])
            stats["Validation %"].append(100 * (val_counts[gesture] / len(self.y_val)))
            stats["Test"].append(test_counts[gesture])
            stats["Test %"].append(100 * (test_counts[gesture] / len(self.y_test)))
        
        stats_df = pd.DataFrame(stats, index=self.gestures)
        print(f"\nDataset Split into Train, Test and Validation Sets:\n")
        print(stats_df)

        return True

    def create_folders(self):
        """
        Create folder structure for train/test/validation sets.
        """

        self.processed_dir = self.raw_img_dir.parent/"processed"
        self.train_dir = self.processed_dir/"train"
        self.test_dir = self.processed_dir/"test"
        self.val_dir = self.processed_dir/"val"

        for split_dir in (self.train_dir, self.val_dir, self.test_dir):
            for gesture in self.gestures:
                gesture_dir = split_dir / gesture
                gesture_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nFolder Structure Created\n")

    
    def copy_files(self):
        """
        Copy images from raw/ to processed/(train/val/test) folders
        """
        print(f"Copying Files...\n")

        #Copy into the training set
        for i, img_path in enumerate(self.X_train):
            src = img_path
            label = self.y_train[i]
            dst = self.train_dir/label/img_path.name
            shutil.copy2(src, dst)
        
        #Copy into the test set
        for i, img_path in enumerate(self.X_test):
            src = img_path
            label = self.y_test[i]
            dst = self.test_dir/label/img_path.name
            shutil.copy2(src, dst)
        
        #Copy into the validation set
        for i, img_path in enumerate(self.X_val):
            src = img_path
            label = self.y_val[i]
            dst = self.val_dir/label/img_path.name
            shutil.copy2(src, dst)
        
        print(f"All Files Copied Successfully!\n")

    def run(self):

        print('='*60 + "\nSTARTING DATASPLITTING PROCESS\n" + '='*60)

        #Get raw data
        self.get_raw_data()

        ret = self.split_data()
        if not ret:
            print(f"FAILURE -> Error splitting data!")
            return False

        self.create_folders()
        self.copy_files()

        print('='*60 + "\nDATASPLITTING COMPLETE!\n")
        print(f"Data Location: {self.processed_dir}")
        print(f"Training Set: {len(self.X_train)} images")
        print(f"Validation Set: {len(self.X_val)} images")
        print(f"Test Set: {len(self.X_test)} images")
        print('='*60)

        return True
