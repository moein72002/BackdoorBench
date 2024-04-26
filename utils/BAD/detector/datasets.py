import os, random, pathlib
from torch.utils.data import Dataset
import re
from copy import copy

CLEAN_LABEL = 'clean'

def extract_models_paths(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        new_paths = [os.path.join(root, file) for file in files if file.endswith('.pt')]
        paths += new_paths
    return paths

def extract_target(filepath):
    match = re.search(r'target(\d)', filepath)
    if match:
        return int(match.group(1))
    return None

def extract_info_from_filepath(filepath):
    target = extract_target(filepath)
    return {
        'path': filepath,
        'target': target
    }

class ModelDataset(Dataset):
    def __init__(self, cleans_folder, bads_folder, model_loader, sample=False, sample_k=5, discards=None):
        self.data = []
        self.model_data_dict = {}
        self.loader = model_loader
        self.bads_data = []
        self.cleans_data = []
        
        if discards is None:
            discards = []
        
        attack_folders = [x for x in os.listdir(bads_folder) if os.path.isdir(os.path.join(bads_folder, x))]
        for attack_folder in attack_folders:
            if attack_folder in discards + [CLEAN_LABEL]:
                continue
            bad_data = [extract_info_from_filepath(model_path) for model_path in \
                extract_models_paths(os.path.join(bads_folder, attack_folder))]
            
            for i in range(len(bad_data)):
                bad_data[i]['attack'] = attack_folder

            if sample:
                bad_data = random.sample(bad_data, sample_k)
            self.bads_data += bad_data
            self.model_data_dict[attack_folder] = bad_data
        
        if CLEAN_LABEL not in discards:
            
            cleans_data = [extract_info_from_filepath(model_path) for model_path in extract_models_paths(cleans_folder)]
            
            if sample:
                cleans_data = random.sample(cleans_data, sample_k*4)
            self.cleans_data = cleans_data
        
        self.data = self.bads_data + self.cleans_data
        self.model_data_dict[CLEAN_LABEL] = self.cleans_data
    
        random.shuffle(self.data)

    def load_model(self, model_data):
        return self.loader(model_data['path'], model_data)

    def get_random_clean_model(self):
        return self.load_model(random.sample(self.cleans_data, 1)[0])
    
    def get_random_bad_model(self, name=None):
        if name is None:
            model_data = random.sample(self.bads_data, 1)[0]
        else:
            model_data = random.sample(self.model_data_dict[name], 1)[0]
        return self.load_model(model_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.load_model(self.data[idx]), int(self.data[idx].get('attack') is None), self.data[idx]
    