from pathlib import Path
from functools import wraps
import datetime
from time import time, sleep
import pickle

import yaml
import pandas as pd

from src.helper.constant import get_proj_dir, get_data_dir, get_pred_csv, METRICS, METRIC_LABELS


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        elapsed = str(datetime.timedelta(seconds=(end-start)))
        print("{} running time: {}".format(f.__name__, elapsed))
        return result
    return wrapper

def get_reg_pred_df(experiment_id_):
    """Get the predicted data frame.
    NOTE, the cells are not filtered by the velocity,
    since prediction for all cells are possible."""
    exp = Experiment(expid_=experiment_id_, type_='regression')
    seq = exp.info['test_seqs'][0]
    csv = get_pred_csv(exp.info['expid'], seq, filter_=False)
    df = pd.read_csv(csv)
    print(df.head)
    return df

def get_reg_pred_raw(experiment_id_):
    """Get the dict with cell ID as key, raw prediction list as values."""
    exp = Experiment(expid_=experiment_id_, type_='regression')
    df = get_reg_pred_df(exp.info['expid'])
    dim = exp.info['map_d']  # prediction dimension
    names = [f"label_pred_raw_{d}" for d in range(dim)]
    pred_raw = {row['ID']: [row[n] for n in names] for _, row in df.iterrows()}

    return pred_raw


def encode_one_hot(label_, dim_):
    """Sparse label to one-hot vector.
       When index is nan, return evenly distributed direction.
    """
    if pd.isna(label_):
        vec = [1.0/dim_] * dim_
    else:
        vec = [0] * dim_
        vec[int(label_)] = 1
    return vec


def get_reg_gt_raw(experiment_id_):
    """Get the dict with cell ID as key, raw ground truth list as values."""
    exp = Experiment(expid_=experiment_id_, type_='regression')
    df = get_reg_pred_df(exp.info['expid'])
    dim = exp.info['map_d']  # prediction dimension
    gt_raw = {row['ID']: encode_one_hot(row['label_gt'], dim) for _, row in df.iterrows()}

    return gt_raw


def get_reg_simu_raw(experiment_id_, num_ratio_, i_ratio_):
    """Get the dict with cell ID as key, simulated probability as values."""
    if i_ratio_ == 0:
        return get_reg_pred_raw(experiment_id_)
    elif i_ratio_ == (num_ratio_-1):
        return get_reg_gt_raw(experiment_id_)
    else:
        ratio_gt = i_ratio_ / (num_ratio_ - 1)
        gt_raw = get_reg_gt_raw(experiment_id_)
        pred_raw = get_reg_pred_raw(experiment_id_)
        simu_raw = dict()
        for sid in pred_raw.keys():
            simu_raw[sid] = [(1 - ratio_gt) * pred + ratio_gt * gt for (pred, gt)
                             in zip(pred_raw[sid], gt_raw[sid])]
        return simu_raw


class Experiment():
    """Store and save experiment information."""
    def __init__(self, expid_=None, info_=None, type_='linking', verbose_=False):
        self.type = type_
        self.init_time = datetime.datetime.now()
        self.verbose = verbose_
        proj_dir = get_proj_dir()
        exp_dir = Path(proj_dir / f"data/wudi_log/{self.type}")

        # Initialize a new experiment.
        if expid_ == None:
            if info_ == None:
                self.info = dict()
            else:
                self.info = info_

            exp_dir.mkdir(exist_ok=True, parents=True)
            self.info['expid'] = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            self.save_path = str(exp_dir / f"{self.info['expid']}.yaml")

            # Modify the experiment ID if it exists.
            while Path(self.save_path).exists():
                sleep(1)
                self.info['expid'] = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
                self.save_path = str(exp_dir / f"{self.info['expid']}.yaml")

            if self.verbose:
                print(f"Created an experiment with ID {self.info['expid']}.")

        # Load exisiting experiment.
        else:
            self.save_path = str(exp_dir / f"{expid_}.yaml")
            with open(self.save_path, 'r') as f:
                self.info = yaml.load(f, Loader=yaml.Loader)
                if self.verbose:
                    print(f"Loaded the experiment with ID {self.info['expid']}.")

    def update_info(self, new_info_):
        self.info.update(new_info_)

        return 0

    def save_info(self, save_path_=None, save_time_=True):
        if save_path_ == None:
            save_path = self.save_path
        else:
            save_path = save_path_
            Path(save_path).parent.mkdir(exist_ok=True, parents=True)

        if 'save_time' not in self.info and save_time_:
            self.info['save_time'] = str(datetime.datetime.now() - self.init_time)

        with open(save_path, 'w') as f:
            yaml.dump(self.info, f)

        print(f"Saved experiment to {save_path}.")

        return 0

    def get_expid(self):
        return self.info['expid']

    def get_info(self):
        return self.info

    def get_save_path(self):
        return self.save_path


def make_model_tag(gap_, name_, loss_, d_feat_, max_fr_):
    model_tag = f"dt{gap_}_{name_}_{loss_}_feat{d_feat_}_f{max_fr_}"

    return model_tag


def get_batch_expids(name_):
    """Get batch experiment IDs."""
    data_dir = get_data_dir()
    path = data_dir / f"wudi_log/batch/{name_}.yaml"

    with open(path, 'r') as stream:
        try:
            info = yaml.safe_load(stream)
            expids = info['expids']
        except yaml.YAMLError as exc:
            print(exc)

    return expids

def get_batch_conf(name_, type_):
    """Get representative configuration of batch experiments."""
    expids = get_batch_expids(name_)
    exp = Experiment(expids[0], type_=type_)
    return exp.info

class ExperimentBatch():
    def __init__(self, name_):
        # Get the batch path from name.
        self.name = name_
        self.expids = get_batch_expids(name_)
        self.gap = self.get_exp_gap()
        self.seq_expids = self.get_seq_expids()

    def get_seq_expids(self):
        seq_expids = dict()
        for expid in self.expids:
            exp = Experiment(expid, type_='linking')
            seq = exp.info['pred_seq']
            seq_expids[seq] = exp.info['expid']
        return seq_expids

    def get_exp_gap(self):
        exp = Experiment(self.expids[0], type_='linking')
        gap = exp.info['gap']
        return gap

    def get_expid(self):
        raise NotImplementedError


class RegressionBatch(ExperimentBatch):
    def __init__(self, name_):
        super().__init__(name_)
        self.min_vel = self.get_min_vel()

    def get_seq_expids(self):
        seq_expids = dict()
        for expid in self.expids:
            exp = Experiment(expid, type_='regression')
            seq = exp.info['test_seqs'][0]
            seq_expids[seq] = exp.info['expid']
        return seq_expids

    def get_exp_gap(self):
        exp = Experiment(self.expids[0], type_='regression')
        gap = exp.info['gap']
        return gap

    def get_min_vel(self):
        exp = Experiment(self.expids[0], type_='regression')
        min_vel = exp.info['min_vel']
        return min_vel

    def get_expid(self, seq_):
        expid = self.seq_expids[seq_]
        return expid


def load_pickle(path_):
    with open(path_, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(path_, data_):
    with open(path_, 'wb') as f:
        pickle.dump(data_, f)
    return 0


def print_score(score_, metrics_):
    """Print the score of model evaluation."""
    print(f"Test loss: {score_[0]:.2f}")

    for i, metric in enumerate(metrics_):
        print(f"{metric}: {score_[i+1]:.2f}")
        if metric == 'mse_angle':
            print(f"RMSE angle: {score_[i+1]**0.5*180:.2f} deg")

    return 0


def dict_score(score_, metrics_):
    """Create a dictionary from the score of model evaluation."""
    info = dict()
    info['loss'] = score_[0]
    for i, metric in enumerate(metrics_):
        info[metric] = score_[i+1]
        if metric == 'mse_angle':
            info['rmse_angle_deg'] = score_[i+1]**0.5*180

    return info