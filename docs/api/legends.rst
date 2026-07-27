Legends
=======

Multi-channel legends: one block per visual channel (color, shape, size,
edge, fill, alpha, orientation, line, region, colorbar, glyph, text, or a
custom artist), stacked and placed inside the axes or in the figure margin.
:class:`~skyplothelper.MultiLegend` is the fluent container; the block
classes below are its building blocks and can also be constructed directly.
Narrative and worked examples: :doc:`/guide/legends`. The same blocks render
on the interactive backend via :func:`sphpl.add_legend
<skyplothelper.plotly.add_legend>` (:doc:`/api/plotly`).

.. currentmodule:: skyplothelper

Container
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   MultiLegend

Blocks
------

Each named block varies one channel; :class:`LegendBlock` is the generic
core they all wrap (entries are style dicts + a ``swatch_kind``).
:meth:`SizeBlock.from_catalog` builds a size key that reproduces a
:func:`plot_catalog` result's exact marker sizes.

.. autosummary::
   :toctree: generated
   :nosignatures:

   LegendBlock
   ColorBlock
   ShapeBlock
   SizeBlock
   EdgeBlock
   FillBlock
   AlphaBlock
   OrientBlock
   LineBlock
   RegionBlock
   ColorbarBlock
   GlyphBlock
   TextBlock

Glyph registry
--------------

The shared registry backing :meth:`MultiLegend.add_glyph` /
:class:`GlyphBlock` — the reticle shapes are pre-registered; add your own so
a legend swatch reuses the real glyph geometry.

.. autosummary::
   :toctree: generated
   :nosignatures:

   register_glyph
   list_glyphs
