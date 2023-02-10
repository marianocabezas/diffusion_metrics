import os
import re
import time
import nibabel as nib
from mrtrix import load_mrtrix
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import csv
import numpy as np
import pandas as pd
import seaborn as sn
import statsmodels.api as smapi
from connectome import conn_mask


def generate_data_dict(
    main_path, gt, brain_name='{:}_brainmask.nii.gz',
    wm_name='{:}_wm_mask.mif.gz', fod_name='{:}_wmfod_norm.mif.gz',
    fixel_name='fixels', connectome_tag='connectome_DK'
):
    """
    A certain folder structure is assumed to construct the dictionary for now.
    main_path/
    |-----<subject/>
             |-----<brain_name>
             |-----<wm_name>
             |-----<methods/>
                      |-----<fixel_name/>
                      |-----<fod_name>
                      |-----<connectome_name>
    For the file names, subject id will always be passed as the first parameter
    to the string. While this behavior might be extended, currently we provide
    a minimum set of requirements.
    :param main_path:
    :param gt:
    :param brain_name:
    :param wm_name:
    :param fod_name:
    :param fixel_name:
    :param connectome_tag:
    :return:
    """
    subjects = sorted(os.listdir(main_path))
    methods = sorted(
        f for f in os.listdir(os.path.join(main_path, subjects[0]))
        if os.path.isdir(os.path.join(main_path, subjects[0], f))
    )
    data_dict = {}
    for sub in subjects:
        sub_path = os.path.join(main_path, sub)
        sub_dict = {
            'brain': os.path.join(sub_path, brain_name.format(sub)),
            'wm': os.path.join(sub_path, wm_name.format(sub)),
            'gt': {},
            'methods': {},
        }
        for method in methods:
            method_dict = {
                'fod': os.path.join(
                    sub_path, method, fod_name.format(sub)
                ),
                'fixel': os.path.join(
                    sub_path, method, fixel_name
                ),
                'connectome': find_file(
                    connectome_tag, os.path.join(sub_path, method)
                ),
            }
            if method == gt:
                sub_dict['gt'] = method_dict
            else:
                sub_dict['methods'][method] = method_dict
        data_dict[sub] = sub_dict
    return data_dict


def find_file(name, dirname):
    """

    :param name:
    :param dirname:
    :return:
    """
    result = list(filter(
        lambda x: not os.path.isdir(x) and re.search(name, x),
        os.listdir(dirname)
    ))

    return os.path.join(dirname, result[0]) if result else None


def load_image(image_path):
    if image_path.endswith('.mif.gz') or image_path.endswith('.mif'):
        image = load_mrtrix(image_path).data
    elif image_path.endswith('.nii.gz') or image_path.endswith('.nii'):
        image = nib.load(image_path).get_fdata()
    else:
        image = None
        raise IOError('file extension not supported: ' + str(image_path))

    return image


def load_fixel_data(
    fixel_path, directions='directions.mif', index='index.mif', afd='afd.mif',
    peak='peak.mif'
):
    index_file = os.path.join(fixel_path, index)
    afd_file = os.path.join(fixel_path, afd)
    peak_file = os.path.join(fixel_path, peak)
    dir_file = os.path.join(fixel_path, directions)

    index_tuples = load_mrtrix(index_file).data
    afd_vector = load_mrtrix(afd_file).data.squeeze()
    peak_vector = load_mrtrix(peak_file).data.squeeze()
    dir_matrix = load_mrtrix(dir_file).data.squeeze()

    return index_tuples, afd_vector, peak_vector, dir_matrix


def load_connectome(csv_path, dk_atlas=True):
    conn = np.genfromtxt(
        csv_path, delimiter=','
    )
    if dk_atlas:
        exclude_nodes = [1, 31, 32, 50, 80, 81]
    else:
        exclude_nodes = []
    m = np.min(conn.shape)
    r, c = np.triu_indices(m, 1)
    return conn, conn[r, c]


def time_to_string(time_val):
    """
    Function to convert from a time number to a printable string that
     represents time in hours minutes and seconds.
    :param time_val: Time value in seconds (using functions from the time
     package)
    :return: String with a human format for time
    """

    if time_val < 60:
        time_s = '%ds' % time_val
    elif time_val < 3600:
        time_s = '%dm %ds' % (time_val // 60, time_val % 60)
    else:
        time_s = '%dh %dm %ds' % (
            time_val // 3600,
            (time_val % 3600) // 60,
            time_val % 60
        )
    return time_s


def print_progress(prefix='', step=0, n_steps=1, t_init=None):
    if t_init is not None:
        t_out = time.time() - t_init
        t_eta = (t_out / (step + 1)) * (n_steps - (step + 1))
        time_s = '<{:} - ETA: {:}>'.format(time_to_string(t_out), time_to_string(t_eta))
    else:
        time_s = ''
    percent = 25 * (step + 1) // n_steps
    progress_s = ''.join(['█'] * percent)
    remainder_s = ''.join([' '] * (25 - percent))
    print(' '.join([' '] * 300), end='\r')
    print(
        '\033[K{:} [{:}{:}] {:06d}/{:06d} - {:05.2f}% {:}'.format(
            prefix, progress_s, remainder_s,
            step, n_steps, 100 * (step + 1) / n_steps,
            time_s
        ),
        end='\r'
    )


def plot_correlation(
        x, y, xlabel='Manual', ylabel='Model', verbose=0
):
    results = smapi.OLS(y, smapi.add_constant(x)).fit()

    if verbose > 1:
        print(results.summary())
    plt.title(
        u"R\u00b2 = {:5.3f} ({:5.3f}, {:5.3f})".format(
            results.rsquared,
            results.pvalues[0], results.pvalues[1]
        )
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    sn.scatterplot(x=x, y=y)
    x_plot = np.linspace(0, np.round(np.max(x)), 1000)
    plt.plot(x_plot, x_plot * results.params[1] + results.params[0], 'k')


def plot_conn_masks(pvalues, edge_list, pipe, prmin=0.95, prmax=1):
    edges = len(pvalues)
    nodes = int((np.sqrt(8 * edges + 1) + 1) / 2)
    r, c = np.triu_indices(nodes, 1)

    mask = conn_mask(edge_list, edges)
    conn = np.zeros((nodes, nodes))
    valid_pvalues = np.zeros(edges)
    valid_pvalues[edge_list] = pvalues[edge_list]
    conn[r, c] = valid_pvalues
    conn[c, r] = valid_pvalues

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sn.heatmap(
        mask, xticklabels=False, yticklabels=False,
        square=True, cmap='jet', vmin=0, vmax=1
    )
    plt.ylabel(pipe)
    plt.subplot(1, 2, 2)
    sn.heatmap(
        conn, xticklabels=False, yticklabels=False,
        square=True, cmap='jet', vmin=prmin, vmax=prmax
    )
