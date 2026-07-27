Ticks & labels
==============

Tick label text and placement. The *formatters* produce label strings —
sexagesimal RA/Dec in configurable separator styles, offset coordinates,
and the anchored-offset (absolute anchor + relative offsets) convention. The
*application* helpers wire a formatter onto an existing frame in one
call and handle the WCSAxes-specific plumbing (tick selection, anchor
alignment) for you. Most users only need
:func:`~skyplothelper.format_ticklabels` and, for zoomed fields, one of
the ``apply_*`` functions. Narrative, with the grid and
coordinate-overlay machinery these pair with: :doc:`/guide/ticks`.

.. currentmodule:: skyplothelper

Formatters
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   format_ticklabels
   format_WCS_ticklabels
   format_mpl_ticklabels
   RAlabelformatter
   RAlabellist
   OffsetFormatter
   AnchoredOffsetFormatter

Tick application
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   apply_offset_ticks
   apply_anchored_offset
   add_curved_lon_ticks
   auto_size_ticklabels
