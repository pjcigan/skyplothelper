

def test_beam_psf_inset_inherits_parent_stroke():
    """The psf-inset beam copy inherits the parent's legibility stroke."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse

    import skyplothelper as sph
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)))
    b = sph.Beam((3, 3), bmaj_pix=3, bmin_pix=2, bpa_deg=20,
                 stroke_color="k", stroke_lw=3).add_to(ax)
    b.add_psf_inset(ax, np.random.default_rng(0).random((8, 8)))
    stroked = [p for a in fig.axes for p in a.patches
               if isinstance(p, Ellipse) and p.get_path_effects()]
    assert len(stroked) >= 2, "inset beam did not inherit the stroke"
    plt.close("all")
