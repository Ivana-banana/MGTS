"""Score held-out corridor and opening transects after a frozen metric plan registration."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mgts_nav.plan_alignment import similarity
from mgts_nav.slam import rotation

CELL = .04
X = (np.arange(251)-125)*CELL
Y = (np.arange(151)-75)*CELL


def office(control):
    # Raster in survey coordinates. Exterior marks and interior transects are disjoint.
    xx, yy = np.meshgrid(X, Y)
    stretch = .36*np.maximum(0, 1-abs(xx-2)/2) if control == "stretch" else 0
    wall = ((abs(abs(xx)-4.8) < CELL/2) & (abs(yy) <= 2.8+1e-8))
    wall |= (abs(abs(yy)-2.8) < CELL/2) & (abs(xx) < 4.8)
    span = abs(xx) < 4.8
    jamb = (xx <= -.52+1e-8) | (xx >= .52-1e-8)
    wall |= (abs(yy+1) <= CELL/2) & span & (jamb | (control == "closed"))
    wall |= (abs(yy-1-stretch) <= CELL/2) & span
    if control == "double":
        wall |= (abs(yy-1.28) < CELL/2) & (xx > .8) & (xx < 3)
    known = ~((xx < -2.8) & (xx > -4) & (abs(yy-1) < .6))
    return np.where(~known, 205, np.where(wall, 0, 254)).astype("uint8")


def evaluate(raster, anchors, plan):
    if plan is None:
        return {"known_fraction": float(np.mean(raster != 205)), "geometry": None}
    _, _, linear, shift = similarity(anchors, plan, False)
    # The metric plan has a rotated/translated basis. Convert its fixed test transects
    # to the survey raster using the fit from exterior marks only.
    orient = rotation(.31)
    origin = np.array([3.2, -1.7])
    def sample(x, y):
        plan_xy = np.column_stack(np.broadcast_arrays(x, y)) @ orient.T + origin
        survey = (plan_xy-shift) @ linear
        ix = np.rint(survey[:, 0]/CELL).astype(int)+125
        iy = np.rint(survey[:, 1]/CELL).astype(int)+75
        return raster[iy, ix]
    errors, spacing, unobserved, locations = [], [], 0, []
    for x in X[(abs(X) <= 4.4) & (abs(X) >= .64)]:
        walls = []
        for center in (-1, 1):
            ys = center+np.arange(-12, 13)*CELL
            line = sample(np.full(len(ys), x), ys)
            if np.any(line == 205):
                unobserved += 1
                walls.append(None)
                continue
            hits = ys[line == 0]
            if not len(hits):
                raise ValueError("Fully viewed transect lacks a wall inside the measurement band")
            deviation = float(abs(hits-center).max())
            errors.append(deviation)
            locations.append([float(x), float(hits[np.argmax(abs(hits-center))])])
            walls.append(hits)
        if all(w is not None for w in walls):
            spacing.append(float(np.max(abs(walls[1][:, None]-walls[0]-2))))
    path = X[abs(X) < 1.5]
    door = sample(path, np.full(len(path), -1.))
    if np.any(door == 205):
        width = None
    elif door[len(door)//2] == 0:
        width = 0.
    else:
        edge = path[door == 0]
        width = float(edge[edge > 0].min()-edge[edge < 0].max())
    return dict(wall_max_m=max(errors), wall_p95_m=float(np.quantile(errors,.95)),
                corridor_spacing_max_error_m=max(spacing), opening_width_m=width,
                opening_error_m=None if width is None else abs(width-1.04),
                coverage=len(errors)/(len(errors)+unobserved), unknown_transects=unobserved,
                worst_xy=locations[int(np.argmax(errors))])


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    anchors = np.array([[-4.8,-2.8],[4.8,-2.8],[-4.8,2.8]])
    plan = anchors @ rotation(.31).T + [3.2,-1.7]
    results = {}
    canvas = Image.new("RGB",(2*502,2*338),"white")
    draw = ImageDraw.Draw(canvas)
    for i, control in enumerate(("correct","stretch","double","closed")):
        raster = office(control)
        row = evaluate(raster, anchors, plan)
        results[control] = row
        x,y = (i%2)*502,(i//2)*338
        canvas.paste(Image.fromarray(np.flipud(raster)).resize((502,302), Image.Resampling.NEAREST), (x,y+25))
        worst = [0,-1] if row["opening_error_m"] > row["wall_max_m"] else row["worst_xy"]
        px, py = x+2*(125+worst[0]/CELL), y+25+2*(75-worst[1]/CELL)
        if max(row["opening_error_m"], row["wall_max_m"]) > CELL:
            draw.ellipse((px-8,py-8,px+8,py+8),outline="red",width=3)
        draw.text((x+8,y+5),control, fill="black")
    results["no_plan"] = evaluate(office("correct"), anchors, None)
    if not (results["correct"]["wall_max_m"] < CELL and results["stretch"]["wall_max_m"] > .3
            and results["double"]["wall_max_m"] > .25 and results["closed"]["opening_width_m"] == 0):
        raise AssertionError("Expected geometric control was not detected")
    canvas.save(output/"geometry.png")
    (output/"scores.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
