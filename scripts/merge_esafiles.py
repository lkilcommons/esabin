from esabin.esafile import EsagridFile
import os,argparse

parser = argparse.ArgumentParser(description="Merge EsagridFile HDF5s")


parser.add_argument("dest_h5fn",
                        type=str,
                        help='HDF5 file to write to',
                        default=None)
parser.add_argument("src_h5fn",
                        type=str,
                        help='HDF5 file to read from',
                        default=None)
parser.add_argument("--append",
                        action='store_true',
                        help="Allow appending to existing destination file",
                        default=False)

args = parser.parse_args()

if not os.path.exists(args.src_h5fn):
    raise IOError('No such source file {}'.format(ars.src_h5fn))

src_hd5f_localdir,src_hd5f_filename = os.path.split(args.src_h5fn)
src_esagrid_file = EsagridFile(src_hd5f_filename,
                                hdf5_localdir=src_hd5f_localdir)

if os.path.exists(args.dest_h5fn):
    if not args.append:
        raise IOError(('Destination file {} exists'.format(args.dest_h5fn))
                      +' (--append to add to this file)')
    dest_hd5f_localdir,dest_hd5f_filename = os.path.split(args.dest_h5fn)
    dest_esagrid_file = EsagridFile(args.dest_h5fn,
                                    hdf5_localdir=dest_hd5f_localdir)


    dest_esagrid_file.append_existing(src_esagrid_file)

else:
    print("Creating new EsagridFile at {} from {}".format(args.dest_h5fn,
                                                          args.src_h5fn))

    dest_esagrid_file = EsagridFile.copy_of_existing(src_esagrid_file,
                                                     args.dest_h5fn)
