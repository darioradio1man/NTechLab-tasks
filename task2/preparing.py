import os
import shutil
import re

data_path = "/home/dario/PycharmProjects/NTechLab-tasks/task2/internship_data"
files = os.listdir(data_path)


def train_maker(name):
    train_dir = f"{data_path}/train/{name}"
    for f in files:
        search_object = re.search(name, f)
        if search_object:
            shutil.move(f'{data_path}/{name}', train_dir)


train_maker("female")
train_maker("male")

female_train = data_path + "/train/female/"
female_val = data_path + "/val/female/"
male_train = data_path + "/train/male/"
male_val = data_path + "/val/male/"

try:
    os.makedirs(female_val)
    os.makedirs(male_val)
except OSError:
    print("Creation of the directory %s failed")
else:
    print("Successfully created the directory %s")

female_files = os.listdir(female_train)
male_files = os.listdir(male_train)

for f in female_files:
    validation_female_search_object = re.search("1\d\d\d\d", f)
    if validation_female_search_object:
        shutil.move(f'{female_train}/{f}', female_val)

for f in male_files:
    validation_male_search_object = re.search("1\d\d\d\d", f)
    if validation_male_search_object:
        shutil.move(f'{male_train}/{f}', male_val)
