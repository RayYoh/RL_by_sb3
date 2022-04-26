import seaborn as sns
sns.set()

import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np
import json

from stable_baselines3.common.monitor import LoadMonitorResultsError, get_monitor_files
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func

"""
假设文件保存的路径格式为：
./data/env_algo/algo_seed/monitor.csv
此格式可在训练时进行更改
"""

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def plot_data(data, x_axis='timesteps', y_axis="r", condition="Condition1", **kwargs):
    
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid", font_scale=1.5)

    if x_axis == X_TIMESTEPS or x_axis==X_EPISODES:
        sns.lineplot(data=data, x='l', y=y_axis, hue=condition, ci='sd', **kwargs)
    elif x_axis == X_WALLTIME:
        sns.lineplot(data=data, x='t', y=y_axis, hue=condition, ci='sd', **kwargs)
    else:
        raise NotImplementedError

    # sns.tsplot(data=data, time=x_axis, value=y_axis, unit="Unit", condition=condition, ci='sd', **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=x_axis, y=y_axis, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc='best').set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data['l'])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

def load_results(path: str) -> pd.DataFrame:
    """
    Load all Monitor logs from a given directory path matching ``*monitor.csv``

    :param path: the directory path containing the log file(s)
    :return: the logged data
    """
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form found in {path}")
    data_frames, headers = [], []
    for file_name in monitor_files:
        with open(file_name, "rt") as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pd.read_csv(file_handler, index_col=None)
            headers.append(header)
            data_frame["t"] += header["t_start"]
        data_frames.append(data_frame)
    data_frame = pd.concat(data_frames)
    data_frame.sort_values("t", inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame["t"] -= min(header["t_start"] for header in headers)
    return data_frame, header["env_id"]

def get_datasets(logdir, condition=None, max_timesteps=None, x_axis='timesteps', y_axis="r", smooth=1):
    """
    Recursively look through logdir for output files 

    Assumes that any file "monitor.csv" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        """
        遍历当前路径
        第一个循环输出root为当前路径，files为当前路径下的文件
        第二个循环输出root为当前路径的子文件夹，files为子文件夹下的文件
        依次
        """
        if len(get_monitor_files(root)) > 0:
            """
            判断是否存在monitor.csv
            """
            exp_name = None
            
            try:
                exp_data, exp_name = load_results(root)
                """
                按照路径，读取算法；从monitor中读取实验名称
                最终legend形式为 env_algo
                """
                last_dir =root.split('\\')[-1]
                algo = last_dir.split('_')[0]
                exp_name += '_'+algo
            except LoadMonitorResultsError:
                print('Could not read from %s'%os.path.join(root,'monitor.csv'))
                continue

            """
            对数据分区，代码源自spinningup画图脚本
            """
            if max_timesteps is not None:
                exp_data = exp_data[exp_data.l.cumsum() <= max_timesteps]
            x, _ = ts2xy(exp_data, x_axis)
            y = np.array(exp_data[y_axis])
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


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None, max_timesteps=None, x_axis='timesteps', y_axis="r", smooth=1):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            """
            将当前路径加入logdirs
            It's for a abstract dir, and append current logdir.
            """
            logdirs += [logdir]
        else:
            """
            将当前路径以及该路径同级文件夹/文件加入logdirs
            """
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    先考虑都是None的情况
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
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
    parser.add_argument('--smooth', '-s', help="smooth the curve", type=int, default=5)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)


        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """
    # args.logdir = ['data\\PandaReach'] #data\\CartPole-v1\\Acr
    log_path = args.logdir
    x_axis = {"steps": X_TIMESTEPS, "episodes": X_EPISODES, "time": X_WALLTIME}[args.x_axis]
    x_label = {"steps": "Timesteps", "episodes": "Episodes", "time": "Walltime (in hours)"}[args.x_axis]

    y_axis = {"success": "is_success", "reward": "r"}[args.y_axis]
    y_label = {"success": "Training Success Rate", "reward": "Training Episodic Reward"}[args.y_axis]
    
    make_plots(log_path, args.legend, x_axis, y_axis, x_label, y_label, 
               args.value, args.count, fontsize=args.fontsize,smooth=args.smooth, select=args.select, 
               exclude=args.exclude, estimator=args.est, max_timesteps=args.max_timesteps)

if __name__ == "__main__":
    main()