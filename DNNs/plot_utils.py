import os
import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pipeline_utils
import glob
import warnings
import seaborn as sns
import scipy
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
            if os.path.exists(cache_filename):
                return pkl.load(open(cache_filename, "rb"))
            else:
                folders = [os.path.join(save_folder, sub_folder) for sub_folder in glob.glob(os.path.join(save_folder, glob_pattern))]
        else:
            # check if file already exist
            cache_filename = os.path.join(save_folder, re.sub(r"\{.*?\}", "", template))
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

def create_queries(df, col_list):
    col_to_val = {}
    for col in col_list:
        col_to_val[col] = sorted(df[col].unique())
    query_dict_list = pipeline_utils.generate_parameter_list(col_to_val)
    query_list = []
    for query_dict in query_dict_list:
        query = ""
        for (key, val) in query_dict.items():
            if isinstance(val, str):
                query += f"{key} == '{val}' & "
            elif is_number(val):
                query += f"{key} == {val} & "
            else:
                raise ValueError(f"key {key} with val {val} type {type(val)} is not supported!")
        if query.endswith(" & "):
            query = query[:-len(" & ")]
        if len(query) > 0:
            query_list.append(query)
    return query_list

def plot_results(input_df, plot_var, x, cols_to_plot, query="", use_seed=False, figsize=(3,4), use_log=True, colormap=None, fig_title=None, plus_one=True, xlabel=None, plot_sem=False, marker=None):
    fig_height, fig_width = figsize
    if len(query) > 0:
        df = input_df.query(query)
    else:
        df = input_df
    var_list = sorted(df[plot_var].unique())
    if colormap is None:
        colors = sns.cubehelix_palette(len(var_list))
    else:
        colors = sns.color_palette(colormap, len(var_list))
    fig, axs = plt.subplots(1, len(cols_to_plot))
    fig.set_figheight(fig_height)
    fig.set_figwidth(len(cols_to_plot)*fig_width)
    if len(cols_to_plot) == 1:
        axs = [axs]
    for i, col_dict in enumerate(cols_to_plot):
        if isinstance(col_dict, dict):
            col, col_query = col_dict["col"], col_dict["query"]
        elif isinstance(col_dict, str):
            col, col_query = col_dict, ""
        else:
            raise ValueError(f"col dict {col_dict} must be dict or str!")
        for j, this_var in enumerate(sorted(var_list)):
            vals = df[df[plot_var] == this_var].sort_values(x)
            if len(col_query) > 0:
                vals = vals.query(col_query)
            if not use_seed:
                if plus_one:
                    axs[i].plot(vals[x] + 1, vals[col], label=this_var, color=colors[j], marker=marker) # epoch starts at 0, +1 to plot with log scale
                else:
                    axs[i].plot(vals[x], vals[col], label=this_var, color=colors[j], marker=marker)
            else: # plot with seed
                vals_mean = vals.groupby(by=x).agg({col: lambda x: x.mean()}).reset_index().sort_values(by=x)
                vals_sem = vals.groupby(by=x).agg({col: lambda x: x.sem()}).reset_index().sort_values(by=x)
                mean = vals_mean[col].values
                sem = vals_sem[col].values
                if plus_one:
                    axs[i].plot(vals_mean[x] + 1, mean, label=this_var, color=colors[j], marker=marker)
                    if plot_sem:
                        axs[i].fill_between(vals_sem[x] + 1, mean - sem, mean + sem, alpha=0.3, color=colors[j])
                else:
                    axs[i].plot(vals_mean[x], mean, label=this_var, color=colors[j], marker=marker)
                    if plot_sem:
                        axs[i].fill_between(vals_sem[x], mean - sem, mean + sem, alpha=0.3, color=colors[j])

        if isinstance(col_dict, dict):
            ax_title = ", ".join(list(col_dict.values()))
        elif isinstance(col_dict, str):
            ax_title = col_dict
        axs[i].set_title(ax_title)
        if xlabel is None:
            axs[i].set_xlabel(x)
        else:
            axs[i].set_xlabel(xlabel)
        # axs[i].grid(True, alpha=0.5)
        axs[i].set_ylim([0, None])
        if use_log:
            axs[i].set_xscale('log')
    # axs[i].legend(bbox_to_anchor=(0.9, -0.2), title=plot_var, title_fontsize=14, ncols=3)
    axs[i].legend(bbox_to_anchor=(1, 1), title=plot_var, title_fontsize=16)
    if fig_title is None:
        if len(query) > 0:
            fig.suptitle(query)
    else:
        fig.suptitle(fig_title)
    plt.tight_layout()
    return fig