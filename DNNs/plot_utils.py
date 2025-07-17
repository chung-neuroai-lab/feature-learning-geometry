import os
import pandas as pd
import pickle as pkl
import numpy as np
from tqdm.notebook import tqdm
import glob
import warnings
import re
from copy import deepcopy

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def convert_to_number(x):
    number = float(x)
    if number.is_integer():
        number = int(number)
    return number
def extract_params(template, filename):
    import parse
    params = parse.parse(template, filename)
    if params is not None:
        params = params.named
        for (key, val) in params.items():
            if is_number(val):
                params[key] = convert_to_number(val)
        return params
    else:
        raise ValueError(f"Template {template} not match with filename {filename}!")

def load_results_to_df(save_folder, template, sub_folders=True, skip_ndarray=True, glob_pattern=None):
    df_list = []
    if sub_folders:
        if glob_pattern is not None:
            # check if file already exist
            cache_filename = os.path.join(save_folder, glob_pattern.replace("*", "") + ".pkl")
            print(f"{cache_filename=}")
            if os.path.exists(cache_filename):
                return pkl.load(open(cache_filename, "rb"))
            else:
                folders = [os.path.join(save_folder, sub_folder) for sub_folder in glob.glob(os.path.join(save_folder, glob_pattern))]
        else:
            # check if file already exist
            cache_filename = os.path.join(save_folder, re.sub(r"\{.*?\}", "", template))
            print(f"{cache_filename=}")
            if os.path.exists(cache_filename):
                return pkl.load(open(cache_filename, "rb"))
            else:
                folders = [os.path.join(save_folder, sub_folder) for sub_folder in os.listdir(save_folder)]
    else:
        # check if file already exist
        cache_filename = os.path.join(save_folder, re.sub(r"\{.*?\}", "", template))
        print(f"{cache_filename=}")
        if os.path.exists(cache_filename):
            return pkl.load(open(cache_filename, "rb"))
        else:
            folders = [save_folder]
    count = 0
    for folder in tqdm(folders):
        # print(f"Processing folder {count}/{len(folders)}")
        count += 1
        files = os.listdir(folder) # use glob.glob(pattern) here instead of os.listdir()
        for filename in files:
            filepath = os.path.join(folder, filename)
            try:
                this_params = extract_params(template, filename)
                file_content = pkl.load(open(filepath, "rb"))
                if len(file_content) == 0:
                    warnings.warn(f"File at {filepath} is empty!")
                if isinstance(file_content, list):
                    for content in file_content:
                        if skip_ndarray:
                            content = {key: val for (key, val) in content.items() if not isinstance(val, np.ndarray)}
                        this_dict = {**this_params, **content}
                        df_list.append(this_dict)
                elif isinstance(file_content, dict):
                    for (key, og_content) in file_content.items():
                        is_shuffled = [False]
                        content = deepcopy(og_content)
                        for shuffle_state in is_shuffled:
                            if isinstance(content, pd.DataFrame):
                                content = content.reset_index()
                                content = content[content["shuffle"] == shuffle_state].mean()
                            content = {content_key: content_val for (content_key, content_val) in content.items()}
                            if skip_ndarray:
                                content = {content_key: content_val for (content_key, content_val) in content.items() if content_val is not None}
                                new_content = {}
                                for (content_key, content_val) in content.items():
                                    if is_number(content_val):
                                        new_content[content_key] = content_val
                                    elif isinstance(content_val, np.ndarray):
                                        try:
                                            new_content[content_key] = content_val.item()
                                        except:
                                            pass
                                content = new_content
                            else:
                                content = {content_key: content_val for (content_key, content_val) in content.items()}
                            this_dict = {**content, "layer": key, **this_params, "shuffled": shuffle_state} # key is layer name. TODO: fix this hard-code
                            df_list.append(this_dict)
                else:
                    raise ValueError(f"File content with type {type(file_content)} not supported!")
            except Exception as e:
                print(e)
    result_df = pd.DataFrame(df_list)
    pkl.dump(result_df, open(cache_filename, "wb"))
    return result_df
