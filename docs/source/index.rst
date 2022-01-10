.. esabin documentation master file, created by
   sphinx-quickstart on Fri Jan  7 15:59:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to esabin's documentation!
==================================

Equal solid angle binning is a way of binning geophysical data which tackles the
problem of bins of fixed latitude/longitude width covering different 
amounts of surface area.

The bins produced by the Equal Solid Angle Scheme each 
cover nearly the same surface area. 

Although many schemes which share this property are possible this
code implements only a fixed latitude width variety 
(that is, all bins have the same latitude span, but each band of 
latitude has a different number of longitudinal bins).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   binning
   storing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
