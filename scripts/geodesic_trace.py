#! /usr/bin/env python3
# Author: UE, 2023

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import numpy as np
import mrcfile
import pymcm.mcm as mcm
from pymcm import emread, emwrite


def main(arg):
    # Parse args
    if (not arg.input_filename.endswith("em")) and (not arg.input_filename.endswith("mrc")):
        raise ValueError("Unknown input image format: {}".format(arg.input_filename))

    if (not arg.output_volume.endswith("em")) and (not arg.output_volume.endswith("mrc")):
        raise ValueError("Unknown output image format: {}".format(arg.output_volume))


    x = [int(x) for x in arg.x.split(',')]
    y = [int(y) for y in arg.y.split(',')]
    if len(x) != 3:
        raise ValueError("End coordinate x must be a 3-element, comma-separated list of integer values.")
    if len(y) != 3:
        raise ValueError("Start coordinate y must be a 3-element, comma-separated list of integer values.")

    # Summary
    print("input speed file = {}".format(arg.input_filename))
    print("output volume = {}".format(arg.output_volume))
    print("output trace = {}".format(arg.output_trace))
    print("x = [{} {} {}]".format(x[0], x[1], x[2]))
    print("y = [{} {} {}]".format(y[0], y[1], y[2]))
    print("maxstep = {}".format(arg.maxstep))
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

    print("dimensions are {} x {} x {}".format(inp.shape[2], inp.shape[1], inp.shape[0]))

    # Process image
    outvol, outtrace = mcm.trace(inp, x, y, maxstep=arg.maxstep, verbose=True, prefer_gpu=arg.use_gpu)

    # Write image output
    if arg.output_volume.endswith('em'):
        emwrite(outvol, arg.output_volume)
    elif arg.output_volume.endswith('mrc'):
        with mrcfile.new(arg.output_volume, overwrite=True) as mrcout:
            mrcout.set_data(outvol)
            mrcout.voxel_size = vs
    else:
        raise ValueError("Unknown output image format: {}".format(arg.output_volume))

    print("")
    print("output image {} successfully written".format(arg.output_volume))

    # Write coord output
    with open(arg.output_trace, 'w') as tfile:
        for i in range(0, outtrace.shape[0]):
            tfile.write("{:.4f}\t{:.4f}\t{:.4f}\n".format(outtrace[i, 0], outtrace[i, 1], outtrace[i, 2]))

    print("output coords {} successfully written\n".format(arg.output_trace))
    print("")
    print("program finished\n")

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Finds the shortest geodesic trace through a binary mask.\n\n'

                                        'Example: geodesic_trace.py -i "volume.mrc" -ov "trace.mrc" -op "trace.txt" -x 23,40,21 -y 54,23,93 ')

    parser.add_argument("-i", "--inputFile", dest="input_filename",
                        help="input .mrc or .em file.", metavar="FILE", type=str, required=True)
    parser.add_argument("-ov", "--outputVol", dest="output_volume",
                        help="output .mrc or .em file.", metavar="FILE", type=str, required=True)
    parser.add_argument("-ot", "--outputTrace", dest="output_trace",
                        help="output trace coordinate file.", metavar="FILE", type=str, required=True)
    parser.add_argument("-y", "--start_point", dest="y",
                        help="voxel coordinate (one-based) of trace start.", metavar="Y1,Y2,Y3", type=str, required=True)
    parser.add_argument("-x", "--end_point", dest="x",
                        help="voxel coordinate (one-based) of trace end.", metavar="X1,X2,X3", type=str, required=True)
    parser.add_argument("-m", "--maxstep", dest="maxstep",
                        help="Maximum number of steps to take before terminating trace.", metavar="STEPS", type=int, required=False, default=10000)
    parser.add_argument("--use_gpu", dest="use_gpu",
                        help="Whether to use CPU or GPU implementation.", metavar="BOOL", type=bool,
                        required=False, default=True)

    args = parser.parse_args()

    sys.exit(main(args))