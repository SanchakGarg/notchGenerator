"""Entities whose geometry is stored in an object coordinate system.

ARC, CIRCLE and LWPOLYLINE are written relative to the entity's extrusion vector, and Fusion
emits an extrusion of (0,0,-1) whenever the flat pattern comes off the far face of the sheet.
In that OCS the x axis is mirrored. LINE and SPLINE have no OCS, so getting this wrong leaves
the outer profile in place while every hole jumps to the wrong side of the part.
"""

from __future__ import annotations

import ezdxf
import ezdxf.path
import numpy as np
import pytest

from conftest import BEND, EXTENT, INTERIOR, OUTER, make_dxf
from notchgen import dxfio, pipeline, render
from notchgen.config import Config
from notchgen.curves import curves_from_entity

FLIPPED = {"extrusion": (0, 0, -1)}


def sampled(entity, eid=0, chord_tol=1e-3) -> np.ndarray:
    return np.vstack([c.flatten(chord_tol) for c in curves_from_entity(entity, eid)])


def centre_of(pts: np.ndarray) -> np.ndarray:
    """Mean of a ring's vertices, not counting the closing point a full circle repeats."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) > 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts.mean(axis=0)


def reference(entity, chord_tol=1e-4) -> np.ndarray:
    """What ezdxf itself says the entity's world-coordinate geometry is."""
    return np.array([(v.x, v.y) for v in ezdxf.path.make_path(entity).flattening(chord_tol)])


def max_deviation(got: np.ndarray, ref: np.ndarray) -> float:
    a, b = ref[:-1], ref[1:]
    ab = b - a
    worst = 0.0
    for p in got:
        ap = p - a
        t = np.clip((ap * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-30), 0, 1)
        worst = max(worst, float(np.min(np.hypot(*(ap - t[:, None] * ab).T))))
    return worst


@pytest.fixture
def msp():
    doc = ezdxf.new("R2000")
    doc.header["$INSUNITS"] = 4
    return doc.modelspace()


@pytest.mark.parametrize("extrusion", [None, (0, 0, -1)])
def test_arc_matches_ezdxf_world_coordinates(msp, extrusion):
    attribs = {} if extrusion is None else {"extrusion": extrusion}
    arc = msp.add_arc((-30.0, 27.0), 4.0, 20.0, 200.0, dxfattribs=attribs)
    assert max_deviation(sampled(arc), reference(arc)) < 2e-3


@pytest.mark.parametrize("extrusion", [None, (0, 0, -1)])
def test_circle_matches_ezdxf_world_coordinates(msp, extrusion):
    attribs = {} if extrusion is None else {"extrusion": extrusion}
    circle = msp.add_circle((-30.0, 27.0), 4.0, dxfattribs=attribs)
    assert max_deviation(sampled(circle), reference(circle)) < 2e-3


@pytest.mark.parametrize("extrusion", [None, (0, 0, -1)])
def test_bulged_lwpolyline_matches_ezdxf_world_coordinates(msp, extrusion):
    attribs = {} if extrusion is None else {"extrusion": extrusion}
    poly = msp.add_lwpolyline(
        [(-33.28, 24.5, 0, 0, -1.0), (-27.37, 29.9, 0, 0, -1.0)],
        close=True,
        dxfattribs=attribs,
    )
    assert max_deviation(sampled(poly), reference(poly)) < 2e-3


def test_a_flipped_hole_is_not_mirrored_onto_the_wrong_side(msp):
    """The regression itself: a hole centred at x=+30 must not land at x=-30."""
    flipped = msp.add_circle((-30.0, 27.0), 4.0, dxfattribs=FLIPPED)
    pts = sampled(flipped)
    centre = centre_of(pts)
    assert centre[0] == pytest.approx(30.0, abs=1e-3)
    assert centre[1] == pytest.approx(27.0, abs=1e-3)
    assert np.hypot(*(pts - centre).T).mean() == pytest.approx(4.0, abs=2e-3)


def test_an_out_of_plane_arc_falls_back_to_world_coordinates(msp):
    """Arbitrary extrusions cannot be an arc in the XY plane, so they are flattened."""
    arc = msp.add_arc((5.0, 5.0), 3.0, 0.0, 180.0, dxfattribs={"extrusion": (0.3, 0.2, 0.93)})
    assert max_deviation(sampled(arc), reference(arc)) < 2e-3


def test_a_block_reference_is_placed_at_its_insertion_point(msp):
    doc = msp.doc
    block = doc.blocks.new("HOLE")
    block.add_circle((0.0, 0.0), 3.0)
    insert = msp.add_blockref("HOLE", (12.0, 34.0))
    pts = sampled(insert)
    centre = centre_of(pts)
    assert centre[0] == pytest.approx(12.0, abs=1e-3)
    assert centre[1] == pytest.approx(34.0, abs=1e-3)


def test_holes_render_on_the_same_side_as_the_profile(tmp_path):
    """End to end: a plate with a flipped hole layer must show the hole inside the outline."""
    path = make_dxf(
        tmp_path,
        [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)],
        [((20.0, 0.0), (20.0, 20.0))],
        [((17.0, 0.0), (17.0, 20.0)), ((23.0, 20.0), (23.0, 0.0))],
    )
    doc = ezdxf.readfile(path)
    # Stored mirrored with a -Z extrusion; the real hole sits at x=+30, inside the plate.
    doc.modelspace().add_circle((-30.0, 10.0), 3.0, dxfattribs={**FLIPPED, "layer": INTERIOR})
    doc.saveas(path)

    mapping = {"outer": OUTER, "interior": INTERIOR, "bend": BEND, "extent": EXTENT}
    result = pipeline.process(str(path), mapping, Config())
    assert result.ok, result.report.text()

    holes = [item for item in result.before if item["role"] == "interior"]
    pts = np.array([p for item in holes for p in item["pts"]])
    assert pts[:, 0].min() > 0.0, "the hole was mirrored outside the plate"
    assert pts[:, 0].max() < 40.0
    assert pts[:, 1].min() > 0.0 and pts[:, 1].max() < 20.0

    # The result view must carry the hole through in the same place.
    after = np.array(
        [p for item in result.after if item["role"] == "interior" for p in item["pts"]]
    )
    assert centre_of(after)[0] == pytest.approx(30.0, abs=1e-3)

    source_circle = next(e for e in result.doc.modelspace() if e.dxftype() == "CIRCLE")
    world_center = source_circle.ocs().to_wcs(source_circle.dxf.center)
    out = tmp_path / "normalized.dxf"
    pipeline.save(result, str(out))
    written_circle = next(e for e in ezdxf.readfile(out).modelspace() if e.dxftype() == "CIRCLE")
    assert tuple(written_circle.dxf.extrusion) == pytest.approx((0.0, 0.0, 1.0))
    assert tuple(written_circle.dxf.center) == pytest.approx(tuple(world_center))


def test_the_sample_holes_sit_where_the_file_says(sample_path):
    """A +Z file must be unaffected by the OCS handling."""
    doc = dxfio.read(sample_path)
    mapping = {"outer": OUTER, "interior": INTERIOR, "bend": BEND, "extent": EXTENT}
    payload = render.document_payload(doc, mapping, 1e-3)
    pts = np.array([p for item in payload if item["role"] == "interior" for p in item["pts"]])
    assert pts[:, 0].min() == pytest.approx(-23.99, abs=0.01)
    assert pts[:, 0].max() == pytest.approx(34.33, abs=0.01)
    assert pts[:, 1].min() == pytest.approx(23.2, abs=0.01)
    assert pts[:, 1].max() == pytest.approx(31.21, abs=0.01)
