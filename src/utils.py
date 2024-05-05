import os

def get_project_root_dir():
    return os.getcwd().split('/miaseg/')[0] + '/miaseg'