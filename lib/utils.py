import os
import glob
import hashlib
import gc
import time
import numpy as np
import requests
import contextlib
from tqdm import tqdm
import torch
import random
from logging import log
from os.path import join as pjoin, exists as pexists

import json
import pickle
import pandas as pd
from .gams.utils import extract_GAM


def download(url, filename, delete_if_interrupted=True, chunk_size=4096):
    """ saves file from url to filename with a fancy progressbar """
    try:
        with open(filename, "wb") as f:
            print("Downloading {} > {}".format(url, filename))
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                total_length = int(total_length)
                with tqdm(total=total_length) as progressbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if data:  # filter-out keep-alive chunks
                            f.write(data)
                            progressbar.update(len(data))
    except Exception as e:
        if delete_if_interrupted:
            print("Removing incomplete download {}.".format(filename))
            os.remove(filename)
        raise e
    return filename


def iterate_minibatches(*tensors, batch_size, shuffle=True, epochs=1,
                        allow_incomplete=True, callback=lambda x:x):
    indices = np.arange(len(tensors[0]))
    upper_bound = int((np.ceil if allow_incomplete else np.floor) (len(indices) / batch_size)) * batch_size
    epoch = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in callback(range(0, upper_bound, batch_size)):
            batch_ix = indices[batch_start: batch_start + batch_size]
            batch = [tensor[batch_ix] for tensor in tensors]
            yield batch if len(tensors) > 1 else batch[0]
        epoch += 1
        if epoch >= epochs:
            break


def process_in_chunks(function, *args, batch_size, out=None, **kwargs):
    """
    Computes output by applying batch-parallel function to large data tensor in chunks
    :param function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...]
    :param args: one or many tensors, each [num_instances, ...]
    :param batch_size: maximum chunk size processed in one go
    :param out: memory buffer for out, defaults to torch.zeros of appropriate size and type
    :returns: function(data), computed in a memory-efficient way
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0: batch_size] for x in args])
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(*output_shape, dtype=first_output.dtype, device=first_output.device,
                          layout=first_output.layout, **kwargs)

    out[0: batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])
    return out


def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


@contextlib.contextmanager
def nop_ctx():
    yield None


def get_latest_file(pattern):
    list_of_files = glob.glob(pattern) # * means all if need specific format then *.csv
    if len(list_of_files) == 0:
        print('No files found!')
        return None
    return max(list_of_files, key=os.path.getctime)


def md5sum(fname):
    """ Computes mdp checksum of a file """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def free_memory(sleep_time=0.1):
    """ Black magic function to free torch memory and some jupyter whims """
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)

def to_float_str(element):
    try:
        return str(float(element))
    except ValueError:
        return element


def seed_everything(seed=None) -> int:
    """
    Borrow it from the pytorch_lightning project
    Function that sets seed for pseudo-random number generators  in:
    pytorch, numpy, python.random and sets PYTHONHASHSEED environment variable.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
        else:
            seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        log.warning(
            f"{seed} is not in bounds, \
            numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    seed = random.randint(min_seed_value, max_seed_value)
    print(f"No correct seed found, seed set to {seed}")
    return seed


def output_csv(the_path, data_dict, order=None, delimiter=','):
    if the_path.endswith('.tsv'):
        delimiter = '\t'

    is_file_exists = os.path.exists(the_path)
    with open(the_path, 'a+') as op:
        keys = list(data_dict.keys())
        if order is not None:
            keys = order + [k for k in keys if k not in order]

        col_title = delimiter.join([str(k) for k in keys])
        if not is_file_exists:
            print(col_title, file=op)
        else:
            old_col_title = open(the_path, 'r').readline().strip()
            if col_title != old_col_title:
                old_order = old_col_title.split(delimiter)

                no_key = [k for k in old_order if k not in keys]
                if len(no_key) > 0:
                    print('The data_dict does not have the '
                          'following old keys: %s' % str(no_key))

                additional_keys = [k for k in keys if k not in old_order]
                if len(additional_keys) > 0:
                    print('WARNING! The data_dict has following additional '
                          'keys %s.' % (str(additional_keys)))
                    col_title = delimiter.join([
                        str(k) for k in old_order + additional_keys])
                    print(col_title, file=op)

                keys = old_order + additional_keys

        vals = []
        for k in keys:
            val = data_dict.get(k, -999)
            if isinstance(val, torch.Tensor) and val.ndim == 0:
                val = val.item()
            vals.append(str(val))

        print(delimiter.join(vals), file=op)


class Timer:
    def __init__(self, name, remove_start_msg=True):
        self.name = name
        self.remove_start_msg = remove_start_msg

    def __enter__(self):
        self.start_time = time.time()
        print('Run "%s".........' % self.name, end='\r' if self.remove_start_msg else '\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_diff = float(time.time() - self.start_time)
        time_str = '{:.1f}s'.format(time_diff) if time_diff >= 1 else '{:.0f}ms'.format(time_diff * 1000)

        print('Finish "{}" in {}'.format(self.name, time_str))


def extract_GAM_from_saved_dir(saved_dir, max_n_bins=256):
    if not pexists(saved_dir):
        with Timer('copying from v'):
            cmd = 'rsync -avzL v:/h/kingsley/node/%s ./logs/' % saved_dir
            print(cmd)
            os.system(cmd)

    assert pexists(saved_dir), 'Either path is wrong or copy fails'

    hparams = json.load(open(pjoin(saved_dir, 'hparams.json')))

    with Timer('Extracting GAMs...'):
        if 'num_trees' in hparams: # NODE model
            return extract_GAM_from_NODE(saved_dir, max_n_bins)
        return extract_GAM_from_baselines(saved_dir, max_n_bins)


def extract_GAM_from_NODE(saved_dir, max_n_bins=256):
    from .trainer import Trainer
    from .data import DATASETS
    assert pexists(pjoin(saved_dir, 'hparams.json')), \
        'No hparams file exists: %s' % saved_dir

    hparams = json.load(open(pjoin(saved_dir, 'hparams.json')))

    assert pexists(pjoin(saved_dir, 'checkpoint_best.pth')), 'No best ckpt exists!'
    model = Trainer.load_best_model_from_trained_dir(saved_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train(False)

    pp = pickle.load(open(pjoin(saved_dir, 'preprocessor.pkl'), 'rb'))

    dataset = DATASETS[hparams['dataset'].upper()](path='./data/')
    all_X = pd.concat([dataset['X_train'], dataset['X_test']], axis=0)

    def predict_fn(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=all_X.columns)

        X = pp.transform(X)
        X = torch.as_tensor(X, device=device)
        with torch.no_grad():
            logits = process_in_chunks(model, X, batch_size=2*hparams['batch_size'])
            logits = check_numpy(logits)

        ret = logits
        if len(logits.shape) == 2 and logits.shape[1] == 2:
            ret = logits[:, 1] - logits[:, 0]
        elif len(logits.shape) == 1: # regression or binary cls
            if pp.y_mu is not None and pp.y_std is not None:
                ret = (ret * pp.y_std) + pp.mu
        return ret

    df = extract_GAM(all_X, predict_fn, max_n_bins=max_n_bins)
    return df


def extract_GAM_from_baselines(saved_dir, max_n_bins=256):
    from .data import DATASETS
    model = pickle.load(open(pjoin(saved_dir, 'model.pkl'), 'rb'))

    hparams = json.load(open(pjoin(saved_dir, 'hparams.json')))

    pp = None
    if pexists(pjoin(saved_dir, 'preprocessor.pkl')):
        pp = pickle.load(open(pjoin(saved_dir, 'preprocessor.pkl'), 'rb'))

    dataset = DATASETS[hparams['dataset'].upper()](path='./data/')
    all_X = pd.concat([dataset['X_train'], dataset['X_test']], axis=0)

    def predict_fn(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=all_X.columns)

        if pp is not None:
            X = pp.transform(X)

        if dataset['problem'] == 'classification':
            prob = model.predict_proba(X)
            return prob[:, 1]
        return model.predict(X)

    predict_type = 'binary_prob' \
        if dataset['problem'] == 'classification' else 'regression'
    df = extract_GAM(all_X, predict_fn, max_n_bins=max_n_bins, predict_type=predict_type)
    return df


def average_GAMs(gam_dirs):
    all_dfs = [extract_GAM_from_saved_dir(pjoin('logs', d)) for d in gam_dirs]

    first_df = all_dfs[0]
    all_ys = [np.concatenate(df.y) for df in all_dfs]
    split_pts = first_df.y.apply(lambda x: len(x)).cumsum()[:-1]
    first_df['y'] = np.split(np.mean(all_ys, axis=0), split_pts)
    first_df['y_std'] = np.split(np.std(all_ys, axis=0), split_pts)
    return first_df


def get_gpu_stat(pitem: str, device_id=0):
    ''' Borrow from pytorch lightning:
        https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.9.0/pytorch_lightning/callbacks/gpu_usage_logger.py#L30-L166
    '''
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={pitem}", "--format=csv,nounits,noheader"],
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
        check=True,
    )

    gpu_usage = [float(x) for x in result.stdout.strip().split(os.linesep)]
    return gpu_usage[device_id]
