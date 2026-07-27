FITS headers
============

Header-level plumbing: read pixel scales and beam parameters out of a
FITS header, convert between pixel and sky coordinates, build minimal
valid headers from scratch, and tame cube headers with degenerate axes
down to the 2-D form the plotting machinery wants.
:func:`~skyplothelper.describe_wcs` prints a readable summary of any
header — the first thing to reach for when a WCS misbehaves. Narrative
context (image display workflows): :doc:`/guide/images`.

.. currentmodule:: skyplothelper

Pixel scales & coordinate grids
-------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   getcdelts
   getdegperpix
   getasecperpix
   getsteradperpix
   getcdmatrix
   header_coord_grids
   convsky2pix
   convpix2sky

Beam parameters
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   beampars_asec_fromhdr
   pixperbeam_from_hdr
   pixperbeam_from_pars

Header construction & reshaping
-------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   makesimpleheader
   force_hdr_to_2D
   force_hdr_to_3D
   force_hdr_floats
   squeeze_image

Diagnostics
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   describe_wcs
   saved_plot_size_reducer
