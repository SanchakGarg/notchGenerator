"""Reading and writing DXF. The only module that touches ezdxf documents directly."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import ezdxf
import ezdxf.bbox
import numpy as np
from ezdxf import blkrefs
from ezdxf.lldxf.tagwriter import TagWriter

from .curves import (
    ArcCurve,
    Curve,
    LineCurve,
    PolylineCurve,
    SplineCurve,
    UnsupportedEntity,
    curves_from_entity,
)
from .report import Report

ROLES = ("outer", "interior", "bend", "extent")

# Onshape writes SHEETMETAL_CUT_LINES, SHEETMETAL_BEND_LINES_UP / _DOWN and — only when
# tangent lines were switched on for the export — SHEETMETAL_BEND_TANGENT_LI, the name cut
# short at 26 characters by Onshape itself.
_PATTERNS = {
    "outer": (r"^outer", r"outer.*profile", r"^out\b", r"profile", r"cut[_\- ]?lines?$"),
    "interior": (r"^interior", r"^inner", r"interior.*profile", r"^hole"),
    "bend": (
        r"^bend$",
        r"^bend[_\- ]?lines?$",
        r"^bends$",
        r"bend[_\- ]?lines?[_\- ](up|down)$",
    ),
    "extent": (r"extent", r"bend.*tangent", r"bend.*zone", r"tangent"),
}

# Roles that may be spread over several layers: Onshape splits bend lines by direction.
_MULTI_LAYER_ROLES = ("bend",)

_ONSHAPE_PREFIX = "SHEETMETAL_"

Mapping = dict[str, "str | list[str] | None"]


def layers_for(mapping: Mapping, role: str) -> list[str]:
    """The layers a role is mapped to. A role is usually one layer but may be several."""
    value = mapping.get(role)
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [v for v in value if v]


def _as_layers(layers: "str | list[str]") -> set[str]:
    return {layers} if isinstance(layers, str) else set(layers)

# Most specific first: "BEND_EXTENT" must be claimed before anything reaches for "BEND",
# and "INTERIOR_PROFILES" before the loose "profile" pattern that finds the outer layer.
_MATCH_ORDER = ("extent", "bend", "interior", "outer")

CURVE_TYPES = ("LINE", "SPLINE", "ARC", "ELLIPSE", "LWPOLYLINE", "POLYLINE", "CIRCLE")

_MM_INSUNITS = {0, 4}  # 0 = unitless, 4 = millimetres


@dataclass
class LayerInfo:
    name: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict:
        return {"name": self.name, "counts": self.counts, "total": self.total}


def layer_census(doc) -> list[LayerInfo]:
    layers: dict[str, LayerInfo] = {}
    for e in doc.modelspace():
        info = layers.setdefault(e.dxf.layer, LayerInfo(e.dxf.layer))
        info.counts[e.dxftype()] = info.counts.get(e.dxftype(), 0) + 1
    return sorted(layers.values(), key=lambda i: i.name)


def suggest_mapping(layers: list[LayerInfo]) -> dict[str, str | None]:
    """Guess which layer plays which role, by name first and by structure afterwards."""
    names = [i.name for i in layers]
    out: Mapping = {r: None for r in ROLES}
    taken: set[str] = set()
    for role in _MATCH_ORDER:
        for pattern in _PATTERNS[role]:
            hits = [
                name
                for name in names
                if name not in taken and re.search(pattern, name, re.IGNORECASE)
            ]
            if not hits:
                continue
            if role in _MULTI_LAYER_ROLES and len(hits) > 1:
                out[role] = hits
                taken.update(hits)
            else:
                out[role] = hits[0]
                taken.add(hits[0])
            break

    # Structure, when the names give nothing away: every bend has exactly two extent
    # lines, so the extent layer holds twice the lines of the bend layer.
    if out["bend"] is None or out["extent"] is None:
        line_only = sorted(
            (i for i in layers if i.name not in taken and set(i.counts) == {"LINE"}),
            key=lambda i: i.total,
        )
        if len(line_only) >= 2 and line_only[1].total == 2 * line_only[0].total:
            for role, info in (("bend", line_only[0]), ("extent", line_only[1])):
                if out[role] is None:
                    out[role] = info.name
                    taken.add(info.name)

    # The outer profile is the biggest thing left; interior profiles are whatever remains.
    leftovers = sorted(
        (i for i in layers if i.name not in taken), key=lambda i: i.total, reverse=True
    )
    if out["outer"] is None and leftovers:
        out["outer"] = leftovers[0].name
        taken.add(leftovers[0].name)
        leftovers = leftovers[1:]
    # An Onshape export keeps its holes on the cut layer alongside the outline, so whatever
    # else is left over there (form marks, countersink symbols) is not interior geometry.
    onshape = any(name.upper().startswith(_ONSHAPE_PREFIX) for name in names)
    if out["interior"] is None and leftovers and not onshape:
        out["interior"] = leftovers[0].name
    return out


def read(path: str):
    return ezdxf.readfile(path)


def check_units(doc, report: Report) -> None:
    insunits = doc.header.get("$INSUNITS", 0)
    if insunits not in _MM_INSUNITS:
        report.warn(
            "units",
            f"$INSUNITS is {insunits}, not millimetres. Depth and tolerances are "
            f"interpreted in the file's own units.",
            insunits=insunits,
        )


def collect_curves(doc, layer: str, report: Report) -> tuple[list[Curve], dict[int, object]]:
    """Curves for one layer, plus the eid -> source entity map used when writing back."""
    curves: list[Curve] = []
    sources: dict[int, object] = {}
    for eid, e in enumerate(doc.modelspace()):
        if e.dxf.layer != layer:
            continue
        sources[eid] = e
        if e.dxftype() not in CURVE_TYPES:
            report.warn(
                "unsupported-entity",
                f"Layer {layer!r} contains a {e.dxftype()}, which carries no profile "
                f"geometry this tool understands; ignored.",
                layer=layer,
                dxftype=e.dxftype(),
            )
            continue
        try:
            pieces = curves_from_entity(e, eid)
        except UnsupportedEntity as exc:
            report.error(
                "unsupported-entity",
                f"Layer {layer!r} contains an unsupported {exc} entity.",
                layer=layer,
                dxftype=str(exc),
            )
            continue
        if len(pieces) > 1 or (pieces and not pieces[0].is_whole):
            report.info(
                "exploded-entity",
                f"A {e.dxftype()} on layer {layer!r} was expanded into "
                f"{len(pieces)} straight and circular pieces; the output will contain those "
                f"pieces rather than the original entity.",
                layer=layer,
                dxftype=e.dxftype(),
            )
        curves.extend(pieces)
    return curves, sources


def segments_on_layer(doc, layer: "str | list[str]", report: Report):
    """Straight segments on a layer (or several), for the bend and extent layers.

    Fusion does not always write these as LINE entities — a bend or extent can arrive as a
    two-vertex LWPOLYLINE, and several of them can share one polyline. Everything is routed
    through the same curve conversion the profile uses, so polylines are expanded into their
    straight runs and object coordinate systems are undone on the way.
    """
    from .bends import Segment

    out = []
    curved = 0
    wanted = _as_layers(layer)
    for eid, e in enumerate(doc.modelspace()):
        if e.dxf.layer not in wanted:
            continue
        layer = e.dxf.layer
        if e.dxftype() not in CURVE_TYPES:
            report.warn(
                "unusable-on-bend-layer",
                f"Layer {layer!r} contains a {e.dxftype()}, which describes no line; ignored.",
                layer=layer,
                dxftype=e.dxftype(),
            )
            continue
        try:
            pieces = curves_from_entity(e, eid)
        except UnsupportedEntity as exc:
            report.warn(
                "unusable-on-bend-layer",
                f"Layer {layer!r} contains an unsupported {exc} entity; ignored.",
                layer=layer,
                dxftype=str(exc),
            )
            continue
        for piece in pieces:
            if isinstance(piece, LineCurve):
                runs = [(piece.a, piece.b)]
            elif isinstance(piece, PolylineCurve):
                runs = list(zip(piece.pts[:-1], piece.pts[1:]))
            else:
                curved += 1
                continue
            for a, b in runs:
                if np.hypot(*(np.asarray(b) - np.asarray(a))) > 0:
                    out.append(Segment(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))
    if curved:
        label = ", ".join(sorted(wanted))
        report.warn(
            "curved-on-bend-layer",
            f"Layer {label!r} contains {curved} curved segment(s). A bend line and its extents "
            f"have to be straight, so those were ignored.",
            layer=label,
            count=curved,
        )
    return out


def _add_curve(msp, curve: Curve, layer: str) -> None:
    attribs = {"layer": layer}
    if isinstance(curve, LineCurve):
        msp.add_line(tuple(curve.a), tuple(curve.b), dxfattribs=attribs)
    elif isinstance(curve, SplineCurve):
        spline = msp.add_spline(dxfattribs=attribs)
        spline.apply_construction_tool(curve.bs)
    elif isinstance(curve, ArcCurve):
        import math

        if abs(curve.sweep) >= 2.0 * math.pi - 1e-9:
            msp.add_circle(tuple(curve.center), curve.radius, dxfattribs=attribs)
            return
        start = math.degrees(curve.start_angle) % 360
        end = math.degrees(curve.start_angle + curve.sweep) % 360
        if curve.sweep < 0:
            start, end = end, start
        msp.add_arc(tuple(curve.center), curve.radius, start, end, dxfattribs=attribs)
    elif isinstance(curve, PolylineCurve):
        msp.add_lwpolyline([tuple(p) for p in curve.pts], dxfattribs=attribs)
    else:  # pragma: no cover - every Curve subclass is handled above
        raise TypeError(f"cannot write {type(curve).__name__}")


def _normalize_circle_ocs(circle) -> bool:
    """Write a -Z circle in +Z OCS without moving its world-coordinate center."""
    if not np.allclose(circle.dxf.extrusion, (0.0, 0.0, -1.0), atol=1e-12):
        return False
    center = circle.ocs().to_wcs(circle.dxf.center)
    thickness = circle.dxf.get("thickness", 0.0)
    circle.dxf.center = center
    circle.dxf.extrusion = (0.0, 0.0, 1.0)
    if thickness:
        circle.dxf.thickness = -thickness
    return True


def _portable_document(source, insunits: int):
    """Copy finished model-space geometry into a clean, broadly supported DXF document."""
    output = ezdxf.new("R2007")
    output.header["$INSUNITS"] = insunits
    output.header["$MEASUREMENT"] = (
        1 if insunits == 4 else source.header.get("$MEASUREMENT", 0)
    )

    source_layers = {layer.dxf.name.casefold(): layer for layer in source.layers}
    layer_names = {entity.dxf.layer for entity in source.modelspace()}
    for name in sorted(layer_names, key=str.casefold):
        if name.casefold() == "0":
            continue
        source_layer = source_layers.get(name.casefold())
        lineweight = source_layer.dxf.get("lineweight", -3) if source_layer else -3
        output.layers.new(
            name,
            dxfattribs={"color": 7, "linetype": "CONTINUOUS", "lineweight": lineweight},
        )

    target = output.modelspace()
    for entity in source.modelspace():
        target.add_foreign_entity(entity)
        copied = target[-1]
        copied.dxf.layer = entity.dxf.layer
        copied.dxf.color = 256
        copied.dxf.linetype = "BYLAYER"
        copied.dxf.lineweight = -1
        if copied.dxftype() == "LWPOLYLINE":
            copied.dxf.const_width = copied.dxf.get("const_width", 0.0)

    if "Defpoints" in output.layers and not any(
        entity.dxf.layer.casefold() == "defpoints" for entity in target
    ):
        output.layers.remove("Defpoints")

    extents = ezdxf.bbox.extents(target)
    if extents.has_data:
        target.dxf.extmin = extents.extmin
        target.dxf.extmax = extents.extmax
        output.header["$EXTMIN"] = extents.extmin
        output.header["$EXTMAX"] = extents.extmax
    return output


def _save_portable_document(doc, out_path: str) -> None:
    """Write after removing ezdxf's private metadata, which LibreCAD also strips."""
    doc.commit_pending_changes()
    doc.classes.add_required_classes(doc.dxfversion)
    doc.update_all()
    if "EZDXF_META" in doc.rootdict:
        doc.rootdict["EZDXF_META"].clear()
        doc.rootdict.remove("EZDXF_META")
    for appid in ("EZDXF", "HATCHBACKGROUNDCOLOR"):
        doc.appids.discard(appid)

    with io.open(
        out_path,
        mode="wt",
        encoding=doc.output_encoding,
        errors="dxfreplace",
    ) as stream:
        tagwriter = TagWriter(stream, dxfversion=doc.dxfversion, write_handles=True)
        doc.export_sections(tagwriter)


def write_result(
    doc,
    mapping: dict[str, str],
    kept: list[Curve],
    added: list[Curve],
    out_path: str,
    report: Report,
    single_layer: bool = True,
) -> None:
    """Write only the notched outer profile and its holes in a clean DXF document."""
    msp = doc.modelspace()
    outer_layer = layers_for(mapping, "outer")[0]
    interior_layer = next(iter(layers_for(mapping, "interior")), None)
    survivors = {c.eid for c in kept if c.is_whole}

    doomed = []
    for eid, e in enumerate(msp):
        layer = e.dxf.layer
        if layer == outer_layer:
            if eid not in survivors:
                doomed.append(e)
        elif interior_layer and layer == interior_layer:
            if single_layer:
                e.dxf.layer = outer_layer
            continue
        else:
            doomed.append(e)
    for e in doomed:
        msp.delete_entity(e)

    for curve in kept:
        if not curve.is_whole:
            _add_curve(msp, curve, outer_layer)
    for curve in added:
        _add_curve(msp, curve, outer_layer)

    normalized_circles = sum(
        _normalize_circle_ocs(e) for e in msp if e.dxftype() == "CIRCLE"
    )
    if normalized_circles:
        report.info(
            "normalized-circles",
            f"Normalized {normalized_circles} circle(s) to positive-Z OCS without moving them.",
            count=normalized_circles,
        )

    if single_layer:
        if outer_layer not in doc.layers:
            doc.layers.add(outer_layer, color=7)
        if interior_layer and interior_layer != outer_layer and interior_layer in doc.layers:
            doc.layers.remove(interior_layer)
    else:
        for role, layer in (("outer", outer_layer), ("interior", interior_layer)):
            if layer and layer not in doc.layers:
                doc.layers.add(layer, color=7 if role == "outer" else 5)
    for layer in (*layers_for(mapping, "bend"), *layers_for(mapping, "extent")):
        if layer in doc.layers and layer not in (outer_layer, interior_layer):
            doc.layers.remove(layer)

    # Orphaned dimension-arrow blocks can be rendered at the origin by some importers.
    unused_blocks = blkrefs.find_unreferenced_blocks(doc)
    for name in sorted(unused_blocks):
        doc.blocks.delete_block(name, safe=False)
    if unused_blocks:
        report.info(
            "purged-blocks",
            f"Removed {len(unused_blocks)} unused block definition(s) from the output.",
            count=len(unused_blocks),
        )

    used_layers = {
        e.dxf.layer.casefold()
        for layout in doc.layouts
        for e in layout
    }
    used_layers.update(
        e.dxf.layer.casefold()
        for block in doc.blocks
        if not block.is_any_layout
        for e in block
    )
    kept_layers = {"0", outer_layer.casefold()}
    if interior_layer:
        kept_layers.add(interior_layer.casefold())
    removed_layers = []
    for layer in list(doc.layers):
        name = layer.dxf.name
        if name.casefold() not in used_layers | kept_layers:
            doc.layers.remove(name)
            removed_layers.append(name)
    if removed_layers:
        report.info(
            "purged-layers",
            f"Removed {len(removed_layers)} unused layer(s) from the output.",
            count=len(removed_layers),
        )

    # Carry only finished model-space geometry forward. A fresh R2007 document
    # removes stale source dictionaries and handles that some strict importers reject.
    source_insunits = doc.header.get("$INSUNITS", 0)
    output_doc = _portable_document(doc, source_insunits)
    output_audit = output_doc.audit()
    if output_audit.errors:
        report.warn(
            "unrepaired-structure",
            f"{len(output_audit.errors)} structural problem(s) remain in the output. First: "
            f"{output_audit.errors[0].message}",
            count=len(output_audit.errors),
        )
    _save_portable_document(output_doc, out_path)
    # Only the basename goes into the report — the full path is a server detail that has no
    # business being shown in the browser.
    report.info(
        "written",
        f"Wrote {Path(out_path).name}: {len(list(output_doc.modelspace()))} entities on "
        f"{len({e.dxf.layer for e in output_doc.modelspace()})} layer(s).",
    )
