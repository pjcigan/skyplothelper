Science modules
===============

Two analysis-flavored subsystems that produce plottable results. The
co-visibility functions answer "what sky can these ground stations see
together?" geometrically (elevation limits only), returning regions
that render and answer membership queries like any other. The
vector-spherical-harmonic functions evaluate and apply the low-degree
VSH basis (rotations, glides, degree-2 terms) used in reference-frame
analysis. Narrative, alongside the vector-field plotters they feed:
:doc:`/guide/vectors`.

.. currentmodule:: skyplothelper

Mutual sky visibility (co-visibility)
-------------------------------------

Geometric mutual-visibility regions for station networks.

.. autosummary::
   :toctree: generated
   :nosignatures:

   covisibility_circles
   covisibility_region
   covisibility_duration_band

Vector spherical harmonics
--------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   vsh_field
   vsh_shift_sources
   vsh_shift_frame
   VSH_PARAM_NAMES
