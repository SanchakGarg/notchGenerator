"""Regression tests against the real Fusion export.

The expected values here were established independently of the implementation, by
evaluating the file's B-splines with De Boor's algorithm and ray-casting the result.
"""

from __future__ import annotations

import ezdxf
import ezdxf.bbox
import pytest

from conftest import MAPPING
from notchgen import dxfio, pipeline
from notchgen.config import Config
from notchgen.loop import build_loops

# Outer-profile entity indices whose geometry the notches consume. All six are the
# near-straight flange-edge splines and the two corner fillets they meet.
EXPECTED_REMOVED = {4, 5, 9, 15, 16, 20}

# One row per notch shoulder: (bend endpoint y, shoulder x, resolved anchor point).
EXPECTED_ANCHORS = {
    (-4.3301, 0.1253): (-4.3301, 0.0000),
    (-9.9850, 0.4962): (-9.9900, 0.1590),
    (-4.3301, 54.2747): (-4.3301, 54.4000),
    (-9.9850, 53.9038): (-9.9850, 54.2424),
    (14.6699, 54.2747): (14.6699, 54.4000),
    (20.3247, 53.9038): (20.3297, 54.2410),
    (14.6699, 0.1253): (14.6699, 0.0000),
    (20.3247, 0.4962): (20.3247, 0.1576),
}


@pytest.fixture
def result(sample_path):
    res = pipeline.process(sample_path, MAPPING, Config())
    assert res.ok, res.report.text()
    return res


def test_layers_are_auto_detected(sample_path):
    found = pipeline.inspect(sample_path)
    assert found.suggested == MAPPING


def test_layer_census(sample_path):
    found = pipeline.inspect(sample_path)
    census = {i.name: i.counts for i in found.layers}
    assert census["OUTER_PROFILES"] == {"LINE": 12, "SPLINE": 10}
    assert census["INTERIOR_PROFILES"] == {"LWPOLYLINE": 2}
    assert census["BEND"] == {"LINE": 2}
    assert census["BEND_EXTENT"] == {"LINE": 4}


def test_outer_profile_is_one_closed_loop(sample_path):
    report = pipeline.Report()
    doc = dxfio.read(sample_path)
    curves, _ = dxfio.collect_curves(doc, "OUTER_PROFILES", report)
    loops, worst_gap = build_loops(curves, Config().stitch_tol)
    assert len(loops) == 1
    assert loops[0].closed
    assert len(loops[0]) == 22
    # Fusion writes full doubles; junction gaps are at rounding level, not at 0.005.
    assert worst_gap < 1e-9
    flipped = {lk.curve.eid for lk in loops[0].links if lk.flipped}
    assert flipped == {5, 9, 16, 20}


def test_four_notches_with_expected_apexes(result):
    assert len(result.notches) == 4
    apexes = sorted((round(n.apex[0], 4), round(n.apex[1], 4)) for n in result.notches)
    assert apexes == [
        (-7.1576, 2.2077),
        (-7.1576, 52.1923),
        (17.4973, 2.2077),
        (17.4973, 52.1923),
    ]


def test_notch_cuts_are_symmetric(result):
    """The part is symmetric, so all four notches must remove the same area."""
    areas = [n.cut_area for n in result.notches]
    assert max(areas) - min(areas) < 5e-3
    assert all(7.0 < a < 7.2 for a in areas)


def test_cut_areas_account_for_the_area_lost(result):
    total_cut = sum(n.cut_area for n in result.notches)
    lost = result.surgery.area_before - result.surgery.area_after
    assert lost == pytest.approx(total_cut, abs=1e-4)


def test_removed_entities_match_expectation(result):
    survivors = {c.eid for c in result.surgery.kept if c.is_whole}
    assert survivors == set(range(22)) - EXPECTED_REMOVED


def test_no_entity_needed_splitting(result):
    """Every anchor snaps to a profile vertex, so no fragment should be emitted."""
    assert all(c.is_whole for c in result.surgery.kept)


def test_anchor_and_shoulder_points(result):
    found = {}
    for notch in result.notches:
        chain = notch.chain
        found[_key(chain[1])] = _key(chain[0])
        found[_key(chain[-2])] = _key(chain[-1])
    assert found == EXPECTED_ANCHORS


def test_every_notch_has_both_stubs(result):
    """The extent ends sit 0.125 and 0.339 inboard of the edge, so stubs are never skipped."""
    for notch in result.notches:
        assert len(notch.chain) == 5


def _key(pt) -> tuple[float, float]:
    return (round(float(pt[0]), 4), round(float(pt[1]), 4))


def test_written_file_is_cut_ready(sample_path, tmp_path):
    res = pipeline.process(sample_path, MAPPING, Config())
    out = tmp_path / "out.dxf"
    pipeline.save(res, str(out), single_layer=False)

    doc = ezdxf.readfile(out)
    layers = {e.dxf.layer for e in doc.modelspace()}
    assert layers == {"OUTER_PROFILES", "INTERIOR_PROFILES"}
    assert "BEND" not in doc.layers
    assert "BEND_EXTENT" not in doc.layers

    kinds: dict[str, int] = {}
    for e in doc.modelspace():
        if e.dxf.layer == "OUTER_PROFILES":
            kinds[e.dxftype()] = kinds.get(e.dxftype(), 0) + 1
    # 12 original lines plus 4 notches x 4 segments; 10 splines less the 6 consumed.
    assert kinds == {"LINE": 28, "SPLINE": 4}

    report = pipeline.Report()
    curves, _ = dxfio.collect_curves(doc, "OUTER_PROFILES", report)
    loops, gap = build_loops(curves, Config().stitch_tol)
    assert len(loops) == 1 and loops[0].closed
    assert gap < 1e-9

    holes, _ = dxfio.collect_curves(doc, "INTERIOR_PROFILES", report)
    hole_loops, _ = build_loops(holes, Config().stitch_tol)
    assert len(hole_loops) == 2
    assert all(lp.closed for lp in hole_loops)
    assert all(lp.area() == pytest.approx(50.311, abs=0.01) for lp in hole_loops)


def test_write_result_defaults_to_a_single_layer(sample_path, tmp_path):
    res = pipeline.process(sample_path, MAPPING, Config())
    out = tmp_path / "out.dxf"
    pipeline.save(res, str(out))

    doc = ezdxf.readfile(out)
    assert {e.dxf.layer for e in doc.modelspace()} == {"OUTER_PROFILES"}
    assert "INTERIOR_PROFILES" not in doc.layers

    report = pipeline.Report()
    holes, _ = dxfio.collect_curves(doc, "OUTER_PROFILES", report)
    loops, _ = build_loops(holes, Config().stitch_tol)
    hole_loops = [lp for lp in loops if lp.area() == pytest.approx(50.311, abs=0.01)]
    assert len(hole_loops) == 2


def test_written_file_has_valid_modelspace_extents(sample_path, tmp_path):
    res = pipeline.process(sample_path, MAPPING, Config())
    out = tmp_path / "out.dxf"
    pipeline.save(res, str(out))

    doc = ezdxf.readfile(out)
    extents = ezdxf.bbox.extents(doc.modelspace())
    assert tuple(doc.header["$EXTMIN"]) == pytest.approx(tuple(extents.extmin))
    assert tuple(doc.header["$EXTMAX"]) == pytest.approx(tuple(extents.extmax))


def test_surviving_splines_are_untouched(sample_path, tmp_path):
    res = pipeline.process(sample_path, MAPPING, Config())
    out = tmp_path / "out.dxf"
    pipeline.save(res, str(out))

    def spline_signatures(doc):
        return {
            tuple(round(float(c), 12) for p in e.control_points for c in (p[0], p[1]))
            for e in doc.modelspace()
            if e.dxftype() == "SPLINE" and e.dxf.layer == "OUTER_PROFILES"
        }

    source = spline_signatures(ezdxf.readfile(sample_path))
    written = spline_signatures(ezdxf.readfile(out))
    assert len(source) == 10
    assert len(written) == 4
    assert written <= source


def test_rerunning_does_not_cut_twice(sample_path, tmp_path):
    res = pipeline.process(sample_path, MAPPING, Config())
    once = tmp_path / "once.dxf"
    pipeline.save(res, str(once))
    # The bend layers are gone, so a second pass has nothing to do and must say so
    # rather than damaging the profile.
    again = pipeline.process(str(once), MAPPING, Config())
    assert not again.ok
    assert {d.code for d in again.report.errors} & {"unknown-layer", "missing-role"}
