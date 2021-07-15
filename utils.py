# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
from __future__ import print_function

import os
import h5py
import json
import numpy as np
# import tensorflow as tf



def write_data(data_fname, data_dict, use_json=False, compression=None):
    """Write data in HD5F format.

    Args:
      data_fname: The filename of teh file in which to write the data.
      data_dict:  The dictionary of data to write. The keys are strings
        and the values are numpy arrays.
      use_json (optional): human readable format for simple items
      compression (optional): The compression to use for h5py (disabled by
        default because the library borks on scalars, otherwise try 'gzip').
    """

    dir_name = os.path.dirname(data_fname)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if use_json:
        the_file = open(data_fname, 'w')
        json.dump(data_dict, the_file)
        the_file.close()
    else:
        try:
            with h5py.File(data_fname, 'w') as hf:
                for k, v in data_dict.items():
                    clean_k = k.replace('/', '_')
                    if clean_k is not k:
                        print('Warning: saving variable with name: ',
                              k, ' as ', clean_k)
                    else:
                        print('Saving variable with name: ', clean_k)
                    hf.create_dataset(clean_k, data=v, compression=compression)
        except IOError:
            print("Cannot open %s for writing.", data_fname)
            raise


def read_data(data_fname):
    """ Read saved data in HDF5 format.

    Args:
      data_fname: The filename of the file from which to read the data.
    Returns:
      A dictionary whose keys will vary depending on dataset (but should
      always contain the keys 'train_data' and 'valid_data') and whose
      values are numpy arrays.
    """

    try:
        with h5py.File(data_fname, 'r') as hf:
            data_dict = {k: np.array(v) for k, v in hf.items()}
            return data_dict
    except IOError:
        print("Cannot open %s for reading." % data_fname)
        raise


def write_datasets(data_path, data_fname_stem, dataset_dict, compression=None):
    """Write datasets in HD5F format.

    This function assumes the dataset_dict is a mapping ( string ->
    to data_dict ).  It calls write_data for each data dictionary,
    post-fixing the data filename with the key of the dataset.

    Args:
      data_path: The path to the save directory.
      data_fname_stem: The filename stem of the file in which to write the data.
      dataset_dict:  The dictionary of datasets. The keys are strings
        and the values data dictionaries (str -> numpy arrays) associations.
      compression (optional): The compression to use for h5py (disabled by
        default because the library borks on scalars, otherwise try 'gzip').
    """

    full_name_stem = os.path.join(data_path, data_fname_stem)
    for s, data_dict in dataset_dict.items():
        write_data(full_name_stem + "_" + s,
                   data_dict, compression=compression)


def read_datasets(data_path, data_fname_stem):
    """Read dataset sin HD5F format.

    This function assumes the dataset_dict is a mapping ( string ->
    to data_dict ).  It calls write_data for each data dictionary,
    post-fixing the data filename with the key of the dataset.

    Args:
      data_path: The path to the save directory.
      data_fname_stem: The filename stem of the file in which to write the data.
    """

    dataset_dict = {}
    fnames = os.listdir(data_path)

    print ('loading data from ' + data_path + ' with stem ' + data_fname_stem)
    for fname in fnames:
        if fname.startswith(data_fname_stem):
            data_dict = read_data(os.path.join(data_path, fname))
            idx = len(data_fname_stem) + 1
            key = fname[idx:]
            data_dict['data_dim'] = data_dict['train_data'].shape[2]
            data_dict['num_steps'] = data_dict['train_data'].shape[1]

            # Get behaviours data dim and num steps. Can be different than data
            if 'train_behaviours' in data_dict:
                data_dict['behaviour_dataset_dims'] = data_dict['train_behaviours'].shape[2]

            dataset_dict[key] = data_dict

    if len(dataset_dict) == 0:
        raise ValueError("Failed to load any datasets, are you sure that the "
                         "'--data_dir' and '--data_filename_stem' flag values "
                         "are correct?")

    print (str(len(dataset_dict)) + ' datasets loaded')
    return dataset_dict
