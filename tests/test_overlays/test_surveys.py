"""Smoke tests for skyplothelper.overlays.surveys."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from skyplothelper.overlays.surveys import (
    SURVEY_FOOTPRINTS,
    add_survey_footprint,
    list_surveys,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_survey_footprints_nonempty():
    assert len(SURVEY_FOOTPRINTS) > 0
    # Common surveys expected
    expected = {"sdss", "des", "lsst"}
    assert expected.issubset(set(SURVEY_FOOTPRINTS.keys()))


def test_list_surveys_runs(capsys):
    list_surveys()
    out = capsys.readouterr().out
    assert "sdss" in out.lower() or "SDSS" in out


@pytest.mark.parametrize("survey", ["sdss", "des", "lsst"])
def test_add_survey_footprint_smoke(survey):
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    add_survey_footprint(ax, survey=survey)
    fig.canvas.draw()


def test_survey_footprint_accepts_a_stroke():
    """Every other overlay exposed stroke_color/stroke_lw; footprints did
    not, and **kwargs was no substitute (those reach patch properties, not
    path effects)."""
    import skyplothelper as sph
    fig, ax = sph.allsky_figure(projection="AIT")
    patches = sph.add_survey_footprint(ax, "SDSS", stroke_color="w",
                                       stroke_lw=3.0)
    assert patches
    assert all(p.get_path_effects() for p in patches)
    plt.close(fig)


def test_survey_footprint_has_no_stroke_by_default():
    import skyplothelper as sph
    fig, ax = sph.allsky_figure(projection="AIT")
    patches = sph.add_survey_footprint(ax, "SDSS")
    assert patches
    assert not any(p.get_path_effects() for p in patches)
    plt.close(fig)
