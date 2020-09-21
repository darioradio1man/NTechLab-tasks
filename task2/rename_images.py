def rename(directory, new_name):
    import os
    for i, filename in enumerate(os.listdir(directory)):
        os.rename(directory + "/" + filename, directory + "/" + new_name + str(i) + ".jpg")


def move(prev, new):
    import os
    import shutil
    for file in os.listdir(prev):
        shutil.move(prev + "/" + file, new)


male_directory = "/home/dario/PycharmProjects/NTechLab-tasks/task2/internship_data/male"
female_directory = "/home/dario/PycharmProjects/NTechLab-tasks/task2/internship_data/female"
new_directory = "/home/dario/PycharmProjects/NTechLab-tasks/task2/internship_data"
rename(female_directory, "female")
rename(male_directory, "male")
