import os

def get_project_root_dir():
    return os.getcwd().split('/scil-segmentation/')[0] + '/scil-segmentation'