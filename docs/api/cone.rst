Cone frames
===========

Cosmology cone (z–RA wedge) diagrams: an angular sky coordinate opens
the wedge, redshift or distance runs along the radius. These are
purpose-built polar frames — not WCSAxes — with their own plotters
(scatter, lines, hexbin/pcolormesh density), twin radial axes driven by
conversion functions, and label/tick controls for the slanted spines.
:func:`~skyplothelper.make_bowtie_frame` builds the double-sided
(two-cap) variant. Narrative: :doc:`/guide/cone`.

.. currentmodule:: skyplothelper

Frame builders
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_cone_frame
   make_bowtie_frame
   make_twinr

Plotters
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   cone_scatter
   cone_scatter_z
   cone_plot
   cone_hexbin
   cone_pcolormesh

Ticks, labels & helpers
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_minor_rticks
   log_r
   flip_label
   set_label_pad
   get_label_pad
   redshift_to_r
