
esabin.esagrid
--------------

This module defines:

#. A class which represents a single bin
#. A (parent) class which represents any collection of bins with a constant width in latitude
#. A (child) class for bins with a constant size in longitude and latitude
#. A (child) class for equal-solid-angle bins (esabins)

.. autoclass:: esabin.esagrid.EsagridBin
    :members:

.. autoclass:: esabin.esagrid.ConstantLatitudeSpacingGrid
    :members:    

.. autoclass:: esabin.esagrid.ConstantAzimuthalSpacingGrid
    :members:

.. autoclass:: esabin.esagrid.Esagrid
    :members:

esabin.esagridfile
------------------

This module implements an HDF5-based data storage/retrieval scheme for binned data

The EsagridFileBinGroup class represents an HDF5 Group containing data
from one bin.

The EsagridFile class represents one HDF5 file which contains one Group 
for each (populated) bin.

.. autoclass:: esabin.esafile.EsagridFileBinGroup
    :members:

.. autoclass:: esabin.esafile.EsagridFile
    :members:    