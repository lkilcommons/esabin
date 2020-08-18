import pytest
from pytest import approx
import tempfile
import shutil,tempfile,os
import numpy as np
import h5py

from esabin.esagrid import EsagridBin
from esabin.esafile import EsagridFile,EsagridFileBinGroup

def generate_data_in_bin(grid,flatind):
    esagrid_bin = EsagridBin(grid,flatind)
    rng = np.random.default_rng()
    n_pts = int(100)
    t = np.arange(n_pts)
    #esagrid uses np.digitize to decide which bin data falls in
    #np.digitize operates on half-open interval [start,end)
    #rng.uniform also generates data on half-open interval [start,end)
    #this *should* work to generate data for only one bin
    lats = rng.uniform(esagrid_bin['slat'],esagrid_bin['elat'],size=(n_pts,))
    azis = rng.uniform(esagrid_bin['sazi'],esagrid_bin['eazi'],size=(n_pts,))
    data = rng.standard_normal(size=(n_pts,))
    return t,lats,azis,data

#Temporary directory that deletes itself after tests run
@pytest.fixture(scope='module')
def temph5dir(request):

    temph5dir=tempfile.mkdtemp()
    
    def remove_temporary_dir():
        shutil.rmtree(temph5dir)

    request.addfinalizer(remove_temporary_dir)

    return temph5dir

def test_esagrid_file_single_bin(temph5dir):
    #use default esagrid
    hdf5_filename = 'test_single_bin.h5'
    esagrid_file = EsagridFile(hdf5_filename,hdf5_local_dir=temph5dir)
    test_bin_flatind = esagrid_file.grid.n_bins-5
    t,lats,azis,data = generate_data_in_bin(esagrid_file.grid,
                                            test_bin_flatind)
    esagrid_file.bin_and_store(t,lats,azis,data)

    populated_bin_flatinds = [flatind for flatind in esagrid_file._bingroups]
    assert [test_bin_flatind] == populated_bin_flatinds

def test_esagrid_file_copy_from_existsing_classmethod(temph5dir):
    src_hdf5_filename = 'source_for_test_copy_from_existing.h5'
    src_esagrid_file = EsagridFile(src_hdf5_filename,hdf5_local_dir=temph5dir)
    src_populated_bin_flatind = 6
    generate_data_in_bin_args = (src_esagrid_file.grid,src_populated_bin_flatind)
    src_esagrid_file.bin_and_store(*generate_data_in_bin(*generate_data_in_bin_args))

    dest_hdf5_filename = 'destination_for_test_copy_from_existing.h5'
    dest_h5fn = os.path.join(temph5dir,dest_hdf5_filename)
    dest_esagrid_file = EsagridFile.copy_of_existing(src_esagrid_file,dest_h5fn)

    dest_populated_bin_flatind = 7
    generate_data_in_bin_args = (dest_esagrid_file.grid,dest_populated_bin_flatind)
    dest_esagrid_file.bin_and_store(*generate_data_in_bin(*generate_data_in_bin_args))
    
    dest_populated_bin_flatinds = [flatind for flatind in dest_esagrid_file._bingroups]
    assert [src_populated_bin_flatind,dest_populated_bin_flatind] == dest_populated_bin_flatinds

def test_esagrid_file_bin_group_append_from_existing(temph5dir):
    populated_bin_flatind = 5
    
    src_hdf5_filename = 'test_esagrid_file_bin_group_append_from_existing_src.h5'
    src_esagrid_file = EsagridFile(src_hdf5_filename,hdf5_local_dir=temph5dir)
    generate_data_in_bin_args = (src_esagrid_file.grid,populated_bin_flatind)
    src_t,src_lats,src_azis,src_data = generate_data_in_bin(*generate_data_in_bin_args)
    src_esagrid_file.bin_and_store(src_t,
                                   src_lats,
                                   src_azis,
                                   src_data)

    dest_hdf5_filename = 'test_esagrid_file_bin_group_append_from_existing_dest.h5'
    dest_esagrid_file = EsagridFile(dest_hdf5_filename,hdf5_local_dir=temph5dir)
    generate_data_in_bin_args = (dest_esagrid_file.grid,populated_bin_flatind)
    dest_t,dest_lats,dest_azis,dest_data = generate_data_in_bin(*generate_data_in_bin_args)
    dest_esagrid_file.bin_and_store(dest_t+dest_t.size, #so no dataset name collision
                                    dest_lats,
                                    dest_azis,
                                    dest_data)
    dest_esagrid_file.append_existing(src_esagrid_file)

    with h5py.File(os.path.join(temph5dir,dest_hdf5_filename),'r') as h5f:
        ts,datasets,additional_attrs = dest_esagrid_file[populated_bin_flatind].read(h5f)

    assert src_data == approx(datasets[0])
    assert dest_data == approx(datasets[1])
