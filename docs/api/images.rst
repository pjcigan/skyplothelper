Images
======

The image-display stack, in three tiers of convenience: the scaling
primitives (interval selection via the ``clip_*`` family + a stretch,
composable through :func:`~skyplothelper.rescale_image` for arrays or
:func:`~skyplothelper.make_norm` for a matplotlib norm), the one-call
quicklook figures built on them (returning result objects with every
artist exposed for follow-up tweaks), and reprojection of FITS/RGB
imagery onto sky frames. Narrative, including when to prefer the norm
route over pre-scaled arrays: :doc:`/guide/images`.

.. currentmodule:: skyplothelper

Clipping, stretching & normalization
------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   clip_percentile
   clip_sigma
   clip_zscale
   auto_interval
   rescale_image
   rescale_percentile
   make_norm
   adjust_gamma
   auto_stretch
   describe_image
   list_stretches

Quicklook & channel-map figures
-------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quicklook_plot
   quicklook_figure
   quicklook_fits
   QuicklookResult
   channel_map
   ChannelMapResult
   DataCube
   MomentMap
   simpleimageplot
   simpleimage_figure
   SimpleImageResult

Reprojection
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_sky_image
   reproject_background
   reproject_rgb_map
