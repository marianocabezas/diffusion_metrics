import os
import time
import nibabel as nib
from mrtrix import load_mrtrix


def generate_data_dict(
    main_path, gt, brain_name='{:}_brainmask.nii.gz',
    wm_name='{:}_wm_mask.mif.gz', fod_name='{:}_wmfod_norm.mif.gz',
    fixel_name='fixels', connectome_name='{:}_connectome_DK_32dir.csv'
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
    :param connectome_name:
    :return:
    """
    subjects = sorted(os.listdir(main_path))
    methods = sorted(
        f for f in os.listdir(os.path.join(main_path, subjects[0]))
        if os.path.isdir(os.path.join(main_path, subjects[0], f))
    )
    data_dict = {}
    for sub in subjects[:1]:
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
                'connectome': os.path.join(
                    sub_path, method, connectome_name.format(sub)
                ),
            }
            if method == gt:
                sub_dict['gt'] = method_dict
            else:
                sub_dict['methods'][method] = method_dict
        data_dict[sub] = sub_dict
    return data_dict


def load_image(image_path):
    if image_path.endswith('.mif.gz') or image_path.endswith('.mif'):
        image = load_mrtrix(image_path).data
    elif image_path.endswith('.nii.gz') or image_path.endswith('.nii'):
        image = nib.load(image_path).get_fdata()
    else:
        image = None
        raise IOError('file extension not supported: ' + str(image_path))

    return image


def get_fixel_data(
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
    print(
        '\033[K{:} [{:}{:}] {:06d}/{:06d} - {:05.2f}% {:}'.format(
            prefix, progress_s, remainder_s,
            step, n_steps, 100 * (step + 1) / n_steps,
            time_s
        ),
        end='\r'
    )