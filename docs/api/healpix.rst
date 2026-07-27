HEALPix
=======

The plotting side of the HEALPix workflow: bin catalogs into dense or
sparse maps, render maps into any frame the package can build, query
pixels spatially, and change resolution. Functions follow the catalog →
map → plot flow top to bottom; the queries and resolution tools serve
both directions. Requires the ``healpix`` extra (healpy); RING ordering
is the default throughout, with ``nest=`` available everywhere.
Narrative: :doc:`/guide/healpix`.

.. currentmodule:: skyplothelper

Binning
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   bin_data_as_healpix
   bin_data_sparse
   image_to_healpix
   HealpixBins
   sources_to_healpix_bins
   sources_to_healpix_plot
   auto_nside

Plotting
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_healpix_allsky
   healpix_allsky_figure
   plot_healpix_map
   plot_healpix_sparse
   mask_seam_crossing_quads
   HealpixResult

Queries & pixel geometry
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   healpix_circle_query
   healpix_polygon_query
   healpix_pixel_corners
   healpix_to_celestial
   healpix_to_canvas

Smoothing & resolution
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   healpix_smooth
   healpix_upgrade
   healpix_downgrade
   healpix_combine
   nside_from_array
