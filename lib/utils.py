"""All utility functions."""
import os
import shutil
import sys
import types
import io
import pickle
import math
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid
import matplotlib.pyplot as plt
import numpy as np
import ctypes
import fnmatch
import importlib
import inspect
import torch
from PIL import Image
from threading import Thread
from omegaconf import OmegaConf
from distutils.util import strtobool
from mpl_toolkits.mplot3d import Axes3D
from typing import Any, List, Tuple, Union, NamedTuple
import seaborn
seaborn.set_theme()


def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    return cfg


# Functionality to import modules/objects by name, and call functions by name
# ------------------------------------------------------------------------------------------

def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed)."""

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name) # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name) # may raise ImportError
            get_obj_from_module(module, local_obj_name) # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == '':
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """Get the directory path of the module containing the given object name."""
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:
    """Determine whether the given object is a top-level function, i.e., defined at module scope using 'def'."""
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """Return the fully-qualified name of a top-level function."""
    assert is_top_level_function(obj)
    module = obj.__module__
    if module == '__main__':
        module = os.path.splitext(os.path.basename(sys.modules[module].__file__))[0]
    return module + "." + obj.__name__



# URL helpers
# ------------------------------------------------------------------------------------------

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


########## Image Reading ##########


def imread_tensor(fpath):
    """Read to 0-1 tensor in [3, H, W]."""
    x = torch.from_numpy(imread(fpath))
    return x.permute(2, 0, 1).float() / 255.


def imread_pil(fpath, size=None):
    img = Image.open(open(fpath, "rb")).convert("RGB")
    if size is not None:
        img = img.resize(size, resample=Image.Resampling.BILINEAR)
    return img


def imread(fpath):
    img = Image.open(open(fpath, "rb")).convert("RGB")
    return np.asarray(img).copy()


def imwrite(fpath, arr):
    with open(fpath, "wb") as f:
        Image.fromarray(arr).save(f)


    
########## Other ###########


class GeneralThread(Thread):
    """A general thread for running a function."""
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args, self.kwargs = args, kwargs

    def run(self):
        self.res = self.func(*self.args, **self.kwargs)


class AsyncExecutor:
    """Record the merging process as a video.
    """
    def __init__(self):
        self.thread = None

    def __call__(self, func, *args):
        if self.thread is not None:
            self.thread.join()
        self.thread = GeneralThread(func, *args)
        self.thread.start()
    
    def clean_sync(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None


########## Dictionary ##########


def dict_append(dic, val, key1, key2=None, key3=None):
    """Create a list or append to it with at most 3 levels."""
    if key1 and not key2 and not key3:
        if key1 not in dic:
            dic[key1] = []
        dic[key1].append(val)
        return

    if key1 not in dic:
        dic[key1] = {}
    if key1 and key2 and not key3:
        if key2 not in dic:
            dic[key1][key2] = []
        dic[key1][key2].append(val)
        return

    if key2 not in dic[key1]:
        dic[key1][key2] = {}
    if key3 not in dic[key1][key2]:
        dic[key1][key2][key3] = []
    dic[key1][key2][key3].append(val)


def cat_dict(dics, cat_tensor=True, is_stack=True, device='cpu'):
    """Concatenate a dictionary of tensors."""
    keys = dics[0].keys()
    res = {}
    for k in keys:
        l = [d[k] for d in dics]
        if isinstance(l[0], torch.Tensor) and cat_tensor:
            res[k] = torch.stack(l) if is_stack else torch.cat(l)
            device = res[k].device
        elif isinstance(dics[0][k], list):
            res[k] = []
            for d in l:
                res[k].extend(d)
        else:
            res[k] = l

        """
        if isinstance(dics[0][k], torch.Tensor):
            res[k] = torch.stack(l) if is_stack else torch.cat(l)
        """
    for k in keys:
        l = [d[k] for d in dics]
        if (isinstance(l[0], float) or isinstance(l[0], int)) and cat_tensor:
            res[k] = torch.tensor(l, device=device)

    return res
