#! /usr/bin/env python3
# Author: UE, 2023

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction, RawDescriptionHelpFormatter
import sys
import numpy as np
import mrcfile
import pymcm.mcm as mcm
from pymcm import emread, emwrite

class CustomFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    pass

def main(arg):
    # Parse args
    if (not arg.input_filename.endswith("em")) and (not arg.input_filename.endswith("mrc")):
        raise ValueError("Unknown input image format: {}".format(arg.input_filename))

    if (not arg.output_filename.endswith("em")) and (not arg.output_filename.endswith("mrc")):
        raise ValueError("Unknown output image format: {}".format(arg.output_filename))

    # Summary
    print("input file = {}".format(arg.input_filename))
    print("output file = {}".format(arg.output_filename))
    print("iterations = {}".format(arg.iterations))
    print("alpha = {}".format(arg.alpha))
    print("beta = {}".format(arg.beta))
    print("")

    # Get data
    if arg.input_filename.endswith("em"):
        inp = emread(arg.input_filename)

        if inp.dtype is not np.float32:
            inp = inp.astype(np.float32)

        vs = 1

    elif arg.input_filename.endswith("mrc"):
        with mrcfile.open(arg.input_filename, 'r+') as mrc:
            vs = mrc.voxel_size

            if mrc.data.dtype is not np.float32:
                inp = mrc.data.astype(np.float32)
            else:
                inp = mrc.data
    else:
        raise ValueError("Unknown input image format: {}".format(arg.input_filename))

    print("dimensions are {} x {} x {}\n".format(inp.shape[2], inp.shape[1], inp.shape[0]))

    # Process image
    outp = mcm.mcm_levelset(inp, arg.iterations, arg.alpha, arg.beta, verbose=True, prefer_gpu=arg.use_gpu)

    # Write image output
    if arg.output_filename.endswith('em'):
        emwrite(outp, arg.output_filename)
    elif arg.output_filename.endswith('mrc'):
        with mrcfile.new(arg.output_filename, overwrite=True) as mrcout:
            mrcout.set_data(outp)
            mrcout.voxel_size = vs

    print("")
    print("output image {} successfully written".format(arg.output_filename))
    print("")
    print("program finished\n")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=CustomFormatter,
                            description='Smooths a volume using mean curvature and levelset motion.\n\n'
                                        'Parameter alpha determines the strength and direction of the levelset motion '
                                        'component:\n'
                                        '\t-1<=alpha<0 inwards movement => Erosion\n'
                                        '\t0<=alpha<1 outwards movement => Dilation\n\n' 
                                        'Parameter beta determines the strength of mean curvature motion:\n'
                                        '\t0 <= beta <= 1\n\n'
                                        'Example: mcm_levelset.py -i "volume.mrc" -o "volume_smooth.mrc" -p 10 -a 0.5 -b 0.5')
    parser.add_argument("-i", "--inputFile", dest="input_filename",
                        help="input .mrc or .em file.", metavar="FILE", type=str, required=True)
    parser.add_argument("-o", "--outputFile", dest="output_filename",
                        help="output .mrc or .em file.", metavar="FILE", type=str, required=True)
    parser.add_argument("-p", "--iterations", dest="iterations",
                        help="number of iterations.", metavar="ITER", type=int, required=True)
    parser.add_argument("-a", "--alpha", dest="alpha",
                        help="level set motion (along surface normals).", metavar="ALPHA", type=float, required=True)
    parser.add_argument("-b", "--beta", dest="beta",
                        help="mean curvature motion (along surface curvature).", metavar="BETA", type=float, required=True)
    parser.add_argument("--gpu", dest="use_gpu", action=BooleanOptionalAction,
                        help="Whether to use CPU or GPU implementation.", type=bool,
                        required=False, default=True)

    args = parser.parse_args()

    sys.exit(main(args))