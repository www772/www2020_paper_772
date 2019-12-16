import os
from time import time
from DataModule import DataModule

class DataUtil():
    def __init__(self, conf):
        self.conf = conf

    # dualpc
    def initializeTask16Handle(self):
        train_filename = "data/amazon_electronics.train.json"
        val_filename = "data/amazon_electronics.val.json"
        test_filename = "data/amazon_electronics.test.json"

        # following files are prepared for preference prediction model
        review_train_rerpesentation_path = 'data/train_representation.npy'
        review_val_rerpesentation_path = 'data/val_representation.npy'
        review_test_rerpesentation_path = 'data/test_representation.npy'
     
        self.train = DataModule(self.conf, 
            train_filename, 
            review_representation_path=review_train_rerpesentation_path)
        self.val = DataModule(self.conf, 
            val_filename, 
            review_representation_path=review_val_rerpesentation_path)
        self.test = DataModule(self.conf, 
            test_filename, 
            review_representation_path=review_test_rerpesentation_path)