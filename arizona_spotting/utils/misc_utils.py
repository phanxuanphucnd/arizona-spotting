# -*- coding: utf-8 -*-

import os
import urllib.request

from typing import Any
from .progressbar import MyProgressBar

def str2bool(value):
    return str(value).lower() in ('yes', 'true', 't', '1')

def ifnone(a: Any, b: Any) -> Any:
    """a if a is not None, otherwise b."""
    return b if a is None else a

def get_from_registry(key, registry):
    if hasattr(key, 'lower'):
        key = key.lower()

    if key in registry:
        return registry[key]
    else:
        raise ValueError(f"Key `{key}` not supported, available options: {registry.keys()}")

def extract_loudest_section(wav, win_len=30):
    wav_len = len(wav)
    temp = abs(wav)
    st, et = 0, 0
    max_dec = 0

    for ws in range(0, wav_len, win_len):
        cur_dec = temp[ws: ws + 16000].sum()
        if cur_dec >= max_dec:
            max_dec = cur_dec
            st, et = ws, ws + 16000
            
        if ws + 16000 > wav_len:
            break

    return wav[st: et]

def download_url(url:str, dest:str, name:str, overwrite:bool=False):
    """Download the model from `url` and save it to `dest` with the name is `name`

    :param url: The url to download
    :param dest: The directory folder to save
    :param name: The name file to save
    :param overwrite: If True, overwrite the old file.
    """
    if not os.path.exists(dest):
        print(f"Folder {dest} does not exist. Create a new folder in {dest}")
        os.makedirs(dest)

    if os.path.exists(dest + name) and not overwrite:
        return

    print(f"Downloading file from: {url}")
    try:
        urllib.request.urlretrieve(url, dest + name, MyProgressBar(name))
        print(f"Path to the saved file: `{dest + name}`")
    except:
        print(f"ERROR: Cann't download the file `{name}` from: `{url}`")
