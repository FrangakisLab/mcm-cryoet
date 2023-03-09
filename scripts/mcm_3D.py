#! /usr/bin/env python3
# Author: UE, 2023

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
import sys
import numpy as np
import mrcfile
import pymcm.mcm as mcm
from pymcm import emread, emwrite


def main(arg):
    # Parse args
    if (not arg.input_filename.endswith("em")) and (not arg.input_filename.endswith("mrc")):
        raise ValueError("Unknown input image format: {}".format(arg.input_filename))

    if (not arg.output_filename.endswith("em")) and (not arg.output_filename.endswith("mrc")):
        raise ValueError("Unknown output image format: {}".format(arg.output_filename))

    pixelwidth = [float(h) for h in arg.pixel_width.split(',')]
    if len(pixelwidth) != 3:
        raise ValueError("Pixel Width must be a 3-element, comma-separated list of floating point values.")

    hx = pixelwidth[2]
    hy = pixelwidth[1]
    hz = pixelwidth[0]

    # Summary
    print("input file = {}".format(arg.input_filename))
    print("output file = {}".format(arg.output_filename))
    print("iterations = {}".format(arg.iterations))
    print("h = [{} {} {}]".format(hx, hy, hz))
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

    print("dimensions are {} x {} x {}\n".format(inp.shape[2], inp.shape[1], inp.shape[0]))

    # Process image
    outp = mcm.mcm(inp, arg.iterations, hx=hx, hy=hy, hz=hz, verbose=True, prefer_gpu=arg.use_gpu)

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
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            description='Smooths a volume using mean curvature motion.\n\n'
                                        
                                        'Example: mcm_3D -i "volume.mrc" -o "volume_smooth.mrc" -p 10')
    parser.add_argument("-i", "--inputFile", dest="input_filename",
                        help="input .mrc or .em file.", metavar="FILE", type=str, required=True)
    parser.add_argument("-o", "--outputFile", dest="output_filename",
                        help="output .mrc or .em file.", metavar="FILE", type=str, required=True)
    parser.add_argument("-p", "--iterations", dest="iterations",
                        help="number of iterations.", metavar="ITER", type=int, required=True)
    parser.add_argument("-hxyz", "--pixel_width", dest="pixel_width",
                        help="Pixel widths in x, y, z dimensions.", metavar="HX,HY,HZ", type=str, required=False, default='1,1,1')
    parser.add_argument("-g", "--gpu", dest="use_gpu", action=BooleanOptionalAction,
                        help="Whether to use CPU or GPU implementation.", type=bool,
                        required=False, default=True)

    args = parser.parse_args()

    sys.exit(main(args))