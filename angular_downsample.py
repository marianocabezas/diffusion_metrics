import argparse
import numpy as np
from downsampling import extract_single_shell
from utils import load_mrtrix


def parse_inputs():
    parser = argparse.ArgumentParser(
        description='Train models with incremental learning approaches and '
                    'test them to obtain timeseries metrics of simple'
                    'overlap concepts.'
    )

    # Mode selector
    parser.add_argument(
        '-i', '--input-file',
        dest='input',
        help='Path to the input file.'
    )
    parser.add_argument(
        '-o', '--output-file',
        dest='output',
        help='Path to the output file.'
    )
    parser.add_argument(
        '-b', '--bval',
        dest='bval',
        type=int, default=1000, nargs='+',
        help='B value for single shell sampling.'
    )
    parser.add_argument(
        '-B', '--bval-range',
        dest='bval_range',
        type=int, default=20,
        help='Range of the B value.'
    )
    parser.add_argument(
        '-d', '--directions',
        dest='directions',
        type=int, default=32, nargs='+',
        help='Number of evenly distributed directions to sample.'
    )
    parser.add_argument(
        '-r', '--random-b0',
        dest='rand_b0',
        action='store_true', default=False,
        help='Whether to use a random b0 or the mean b0.'
    )

    options = vars(parser.parse_args())

    return options


def main():
    options = parse_inputs()
    dwi_in = options['input']
    dwi_out = options['output']
    directions = options['directions']
    bval = options['bval']
    bval_range = options['bval_range']
    if dwi_in.endswith('.mif') or dwi_in.endswith('.mif.gz'):
        print('Reading the MIF file {:}'.format(dwi_in))
        mif = load_mrtrix(dwi_in)
        old_grad = mif.grad
        bvals = old_grad[:, -1]
        bvecs = old_grad[:, :-1]

        if not isinstance(bval, list):
            if isinstance(directions, list):
                directions = directions[0]
                print(
                    'Warning: Only {:} directions will be extracted'.format(
                        directions
                    )
                )

            print(
                'Downsampling {:d} directions (B = {:d} ± {:d})'.format(
                    directions, bval, bval_range
                )
            )
            lr_bvecs, lr_bvals, lr_index, b0_index = extract_single_shell(
                bvals, bvecs, bval, bval_range, directions
            )
            lr_index = np.array(lr_index.tolist())
            lr_bvals = np.array([0] + lr_bvals.tolist())
            lr_bvecs = np.concatenate([np.zeros((1, 3)), lr_bvecs])
        elif not isinstance(directions, list):
            print(
                'Warning: {:} directions will be extracted '
                'for each bval ([{:}])'.format(
                    directions, ', '.join([str(bval) for bval in bvals])
                )
            )

            print(
                'Downsampling {:d} directions (B = [{:}] ± {:d})'.format(
                    directions, ', '.join([str(bval_i) for bval_i in bval]),
                    bval_range
                )
            )
            index_list = []
            bval_list = [0]
            bvec_list = [np.zeros((1, 3))]
            for bval_i in bval:
                bvecs_i, bvals_i, index_i, b0_index = extract_single_shell(
                    bvals, bvecs, bval_i, bval_range, directions
                )
                index_list.extend(index_i.tolist())
                bval_list.extend(bvals_i.tolist())
                bvec_list.extend(bvecs_i)
            lr_index = np.array(index_list)
            lr_bvals = np.array(bval_list)
            lr_bvecs = np.concatenate(bvec_list)
        else:
            if len(bval) != len(directions):
                print(
                    'Warning: The numbe of bvals ({:})'
                    'and directions ({:}) do not match. '
                    'Using shortest list length.'.format(
                        len(bval), len(directions)
                    )
                )

            print(
                'Downsampling [{:}] directions (B = [{:}] ± {:d})'.format(
                    ', '.join([str(dir_i) for dir_i in directions]),
                    ', '.join([str(bval_i) for bval_i in bval]),
                    bval_range
                )
            )
            index_list = []
            bval_list = [0]
            bvec_list = [np.zeros((1, 3))]
            for bval_i, dir_i in zip(bval, directions):
                bvecs_i, bvals_i, index_i, b0_index = extract_single_shell(
                    bvals, bvecs, bval_i, bval_range, dir_i
                )
                index_list.extend(index_i.tolist())
                bval_list.extend(bvals_i.tolist())
                bvec_list.extend(bvecs_i)
            lr_index = np.array(index_list)
            lr_bvals = np.array(bval_list)
            lr_bvecs = np.concatenate(bvec_list)

        new_grad = np.concatenate([lr_bvecs, lr_bvals.reshape(-1, 1)], axis=-1)

        if options['rand_b0']:
            b0_i = np.random.permutation(b0_index)[0]
            mif_b0 = mif.data[..., b0_i:b0_i + 1]
        else:
            mif_b0 = np.mean(mif.data[..., b0_index], axis=-1, keepdims=True)
        f_data = np.concatenate([mif_b0, mif.data[..., lr_index]], axis=-1)

        mif.data = f_data
        mif.grad = new_grad

        print('Converting to MIF (Downsampled) {:}'.format(dwi_out))
        mif.save(dwi_out)
    else:
        raise IOError('ERROR: Only MIF files are supported')


if __name__ == '__main__':
    main()