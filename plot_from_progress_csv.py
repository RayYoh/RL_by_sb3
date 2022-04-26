import seaborn as sns
sns.set()

import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np
import json
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func
from stable_baselines3.common.logger import read_csv

EXT = "progress.csv"
DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

class EmptyDataError(ValueError):
    """
    Exception that is thrown in `pd.read_csv` (by both the C and
    Python engines) when empty data or header is encountered.
    """

def get_progress_files(path: str) -> List[str]:
    """
    get all the monitor files in the given path

    :param path: the logging folder
    :return: the log files
    """
    return glob(os.path.join(path, EXT))

def read_csv(filename: str, usecols: List=None) -> pd.DataFrame:
    """
    read a csv file using pandas

    :param filename: the file path to read
    :return: the data in the csv
    """
    return pd.read_csv(filename, index_col=None, usecols=usecols, comment="#")

def read_log(path):
    progress_files = get_progress_files(path)
    if len(progress_files) == 0:
        raise EmptyDataError(f"No progress files of the form {EXT} found in {path}")
    data_frame = read_csv(progress_files[0], usecols=["rollout/ep_rew_mean", "time/total_timesteps"])
    data_frame.rename(columns={"rollout/ep_rew_mean":"r", "time/total_timesteps":"timesteps"}, inplace = True)
    data_frame = data_frame.dropna(subset=["r"])
    data_frame.reset_index()
    return data_frame

def make_plots(all_logdirs, legend=None, x_axis=None, y_axis=None, x_label=None, y_label=None,
               values=None, count=False, figsize=[6.4, 4.8], fontsize=14, smooth=1, select=None, exclude=None, 
               estimator='mean', max_timesteps=None):
    data = get_all_datasets(all_logdirs, legend, select, exclude, max_timesteps, x_axis, y_axis, smooth)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        """
        values目前没用，需要一次性画多张不同图时用
        """
        plt.figure(y_label, figsize=figsize)
        plt.title(y_label, fontsize=fontsize)
        plt.xlabel(f"{x_label}", fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plot_data(data, x_axis=x_axis, y_axis=y_axis, condition=condition, estimator=estimator)
    plt.show()
    return data

def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None, max_timesteps=None, x_axis='timesteps', y_axis="r", smooth=1):
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg, max_timesteps, x_axis=x_axis, y_axis=y_axis, smooth=smooth)
    else:
        for log in logdirs:
            data += get_datasets(log, max_timesteps=max_timesteps, x_axis=x_axis, y_axis=y_axis, smooth=smooth)
    return data

def get_datasets(logdir, condition=None, max_timesteps=None, x_axis='timesteps', y_axis="r", smooth=1):
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if len(get_progress_files(root)) > 0:
            exp_name = None
            try:
                exp_data = read_log(root)
                exp_name = root.split('\\')[-2]
                # last_dir = root.split('\\')[-1]
                # algo = last_dir.split('_')[0]
                # exp_name += '_'+algo
            except EmptyDataError:
                print('Could not read from %s'%os.path.join(root,EXT))
                continue

            """
            对数据分区，代码源自spinningup画图脚本
            """
            if max_timesteps is not None:
                exp_data = exp_data[exp_data.l.cumsum() <= max_timesteps]
            y = np.array(exp_data[y_axis])
            x = np.array(exp_data[x_axis])
            if x.shape[0] >= smooth:
                x, y_mean = window_func(x, y, smooth, np.mean)
            if x_axis == X_TIMESTEPS:
                data = pd.DataFrame({'l':x.tolist(), y_axis:y_mean.tolist()})
            elif x_axis == X_EPISODES:
                data = pd.DataFrame({'l':x.tolist(), y_axis:y_mean.tolist()})
            elif x_axis == X_WALLTIME:
                data = pd.DataFrame({'t':x.tolist(), y_axis:y_mean.tolist()})
            else:
                raise NotImplementedError
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            data.insert(len(data.columns),'Unit',unit)
            data.insert(len(data.columns),'Condition1',condition1)
            data.insert(len(data.columns),'Condition2',condition2)
            datasets.append(data)
    return datasets



def plot_data(data, x_axis='timesteps', y_axis="r", condition="Condition1", **kwargs):
    
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=1.5)

    kwargs = {"linewidth":3}
    if x_axis == X_TIMESTEPS or x_axis==X_EPISODES:
        sns.lineplot(data=data, x='l', y=y_axis, hue=condition, ci='sd', **kwargs)
    elif x_axis == X_WALLTIME:
        sns.lineplot(data=data, x='t', y=y_axis, hue=condition, ci='sd', **kwargs)
    else:
        raise NotImplementedError

    plt.legend(loc='best').set_draggable(True)

    xscale = np.max(np.asarray(data['l'])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

def main():
    import argparse
    parser = argparse.ArgumentParser("Gather results, plot training reward/success")
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')

    parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
    parser.add_argument("--fontsize", help="Font size", type=int, default=14)
    parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int)
    parser.add_argument("-x", "--x-axis", help="X-axis", choices=["steps", "episodes", "time"], type=str, default="steps")
    parser.add_argument("-y", "--y-axis", help="Y-axis", choices=["success", "reward"], type=str, default="reward")

    parser.add_argument('--value', help="Which value to plot", default='Performance', nargs='*')
    parser.add_argument('--count', help="average or all", action='store_true')
    parser.add_argument('--smooth', '-s', help="smooth the curve", type=int, default=20)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    
    # args.logdir = ['data\\CartPole-v0_ppo\\',] 
    log_path = args.logdir
    x_axis = {"steps": X_TIMESTEPS, "episodes": X_EPISODES, "time": X_WALLTIME}[args.x_axis]
    x_label = {"steps": "Timesteps", "episodes": "Episodes", "time": "Walltime (in hours)"}[args.x_axis]

    y_axis = {"success": "is_success", "reward": "r"}[args.y_axis]
    y_label = {"success": "Training Success Rate", "reward": "Training Episodic Reward"}[args.y_axis]
    
    make_plots(log_path, args.legend, x_axis, y_axis, x_label, y_label, 
               args.value, args.count, fontsize=args.fontsize,smooth=args.smooth, select=args.select, 
               exclude=args.exclude, estimator=args.est, max_timesteps=args.max_timesteps)

if __name__ == "__main__":
    # csv_files = ['data\\PandaReach-v2_sac\sac_454698830', 'data\\PandaReach-v2_sac\\sac_3257973677', 'data\\PandaReach-v2_sac\\sac_1372634855']
    # csv_files = ['data\\CartPole-v0_ppo\ppo_3567743922', 'data\\CartPole-v0_ppo\ppo_4159629884',]
    # datasets = []
    # for file in csv_files:
    #     df = read_log(file)
    #     datasets.append(df)
    # data_frame = pd.concat(datasets, )
    # data_frame.sort_values("l", inplace=True)
    # data_frame.reset_index(inplace=True)
    # sns.lineplot(data=data_frame, x="l", y="r")
    # plt.show()
    main()