import json
from types import SimpleNamespace

import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from mgts_nav.bag import load_config, point_offsets, reader, scans
from mgts_nav.deskew import deskew_points
from mgts_nav.demo import START_NS, generate
from tools.evaluate_deskew import POINT_DTYPE, sensor_profile, write_bag


EPOCH = 1_700_000_000_000_000_000
SHIFT = 250_000_000
SWEEP_END = 60_000_000


def transcode(source, target, mode, gap):
    store = get_typestore(Stores.ROS2_HUMBLE)
    with reader(source) as bag, Writer(target, version=8) as writer:
        outputs = {c.id: writer.add_connection(c.topic, c.msgtype, typestore=store) for c in bag.connections}
        for c, received, raw in bag.messages():
            msg = bag.deserialize(raw, c.msgtype)
            header = msg.header.stamp
            elapsed = header.sec * 10**9 + header.nanosec - START_NS["inspection"]
            ns = EPOCH + elapsed
            if c.msgtype.endswith("/Odometry") and gap and 4_050_000_000 < elapsed < 4_550_000_000:
                continue
            if c.msgtype.endswith("/PointCloud2"):
                ns += SWEEP_END - SHIFT
                orig = msg.data.view(POINT_DTYPE)
                relative_ns = orig["offset"].astype(np.int64) - SWEEP_END
                absolute = mode == "absolute"
                pts = np.empty(len(orig), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                                 ("offset", "<f8" if absolute else "<i4")])
                for name in "xyz":
                    pts[name] = orig[name]
                if absolute:
                    pts["offset"] = (ns + relative_ns) / 1e9
                    msg.fields[-1].datatype = 8
                else:
                    pts["offset"] = -relative_ns if mode == "backward" else relative_ns
                msg.data = pts.view(np.uint8)
                msg.point_step, msg.row_step = pts.dtype.itemsize, pts.nbytes
            header.sec, header.nanosec = divmod(ns, 10**9)
            writer.write(outputs[c.id], EPOCH + received - START_NS["inspection"],
                         store.serialize_cdr(msg, c.msgtype))


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    path = tmp_path_factory.mktemp("point-time")
    generate(path/"demo", seed=41)
    cfg = sensor_profile()
    write_bag(path/"demo/inspection", path/"source", "inspection", 42, cfg)
    return path, cfg


@pytest.mark.parametrize("gap", [False, True])
def test_three_point_clocks(sweep, gap):
    path, cfg = sweep
    reference = None
    for mode in ("signed", "backward", "absolute"):
        bag = path / f"{mode}-{gap}"
        transcode(path/"source", bag, mode, gap)
        spec = dict(field="offset", unit="s" if mode == "absolute" else "ns")
        if mode != "signed":
            spec["semantics"] = mode
        profile = bag / "sensor.json"
        profile.write_text(json.dumps(dict(cfg, point_time=spec, header_offset_s={"lidar": .25})))
        result = list(scans(bag, load_config(profile)))
        assert len(result) == 100
        assert abs(result[0].t - (EPOCH / 1e9 + .095)) < 3e-7
        rejected = [i for i, scan in enumerate(result) if scan.empty_reason == "deskew_odom_unavailable"]
        assert rejected == (list(range(40, 46)) if gap else [])
        if reference is None:
            reference = result
            print("gap", gap, "rejected", rejected)
            continue
        maximum = 0.
        for first, other in zip(reference, result, strict=True):
            assert (first.t, first.empty_reason) == (other.t, other.empty_reason)
            np.testing.assert_allclose(other.xy, first.xy, atol=2e-6, rtol=0)
            if len(first.xy):
                maximum = max(maximum, np.linalg.norm(other.xy-first.xy, axis=1).max())
        print(mode, "max point error (m)", maximum)


@pytest.mark.parametrize("extra", [{"semantics": "auto"}, {"semantics": "absolute", "unit": "ms"},
                                    {"direction": "backward"}])
def test_invalid_point_time_profile(tmp_path, extra):
    cfg = sensor_profile()
    cfg["point_time"] = dict(dict(field="offset", unit="ns"), **extra)
    path = tmp_path / "sensor.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="point_time"):
        load_config(path)


def test_absolute_time_on_accelerating_platform_and_bad_fields():
    # At local times 1 and 2 s the platform is at x=1 and x=4 m.
    # A wall at x=10 m returns x=9 at acquisition, x=6 in the later reference frame.
    epoch = EPOCH / 1e9
    header = SimpleNamespace(stamp=SimpleNamespace(sec=int(epoch)+1, nanosec=750_000_000))
    field = SimpleNamespace(name="time", count=1, datatype=8, offset=0)
    msg = SimpleNamespace(header=header, fields=[field], height=1, width=1, is_bigendian=False,
                          point_step=8, row_step=8, data=np.array([epoch+.75], dtype="<f8"))
    valid = np.array([True])
    spec = dict(field="time", unit="s", semantics="absolute")
    offsets = point_offsets(msg, spec, valid)
    times = epoch + np.arange(4)
    odom = np.column_stack((times, [0., 1., 4., 9.], np.zeros((4, 2))))
    corrected = deskew_points([[9., 0., 0.]], offsets, epoch+2, odom, 1.)
    np.testing.assert_array_equal(corrected, [[6., 0., 0.]])
    field.datatype = 7
    with pytest.raises(ValueError, match="supported type"):
        point_offsets(msg, spec, valid)
    msg.data = np.array([-1.], dtype="<f4")
    with pytest.raises(ValueError, match="nonnegative"):
        point_offsets(msg, dict(spec, semantics="backward"), valid)
