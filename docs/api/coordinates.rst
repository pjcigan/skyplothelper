Coordinates & math
==================

Array-friendly coordinate utilities used throughout the package and
handy on their own: conversions between the sky coordinate systems
(thin, vectorized wrappers over astropy's machinery), sexagesimal
parsing and formatting, angular separations, and the longitude
wrapping/rescaling math that keeps values in the range a projection
expects. These set the *data* coordinates — the coordinate system of a
frame itself is chosen at construction (:doc:`/guide/concepts`).

.. currentmodule:: skyplothelper

Input normalizers
-----------------

Coerce flexible caller input into the canonical forms the package uses
internally: any sky-coordinate form to ``(lon, lat)`` degrees, and any time
form to an astropy ``Time``.

.. autosummary::
   :toctree: generated
   :nosignatures:

   to_lonlat
   to_time

Frame conversions
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   convert_frame
   icrs_to_galactic
   galactic_to_icrs
   icrs_to_ecliptic
   ecliptic_to_icrs
   galactic_to_ecliptic
   ecliptic_to_galactic
   icrs_to_supergalactic
   supergalactic_to_icrs

Sexagesimal & angle helpers
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   deg2dms
   dms2deg
   deg2hour
   hour2deg
   dec2sex
   sex2dec
   angulardistance
   RAcosDEC_err

Wrapping & rescaling
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   wrap_360
   wrap_pm180
   wrap_pm90
   wrap_pmPI
   wrap_24hr
   wrap_range
   wrap_center_pmrange
   map_to_newrange
   rescale_data_range
