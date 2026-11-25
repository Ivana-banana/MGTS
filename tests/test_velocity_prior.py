import numpy as np
import pytest

from mgts_nav.bag import Scan
from mgts_nav.grid import Grid
from mgts_nav import slam
from mgts_nav.slam import VelocityPrior


def scan(t, xy=None, gyro=0.):
    return Scan(t, np.empty((0, 2)) if xy is None else xy, np.zeros(2), None, gyro)


def test_lateral_prediction_rotates_with_body_and_caps_step():
    prior = VelocityPrior()
    prior.observe(slam.Estimate(0., np.array([0., 0., 0.]), "matched", 1.))
    last = np.array([0., .09, np.pi/2])
    prior.observe(slam.Estimate(.1, last, "matched", 1.))
    prediction = prior.predict(last, None, scan(.25, gyro=.13))
    np.testing.assert_allclose(prediction, [-.135, .09, np.pi/2+.13])
    prior.history.clear()
    prior.observe(slam.Estimate(0., np.zeros(3), "matched", 1.))
    prior.observe(slam.Estimate(.1, np.array([1., 0., 0.]), "matched", 1.))
    assert prior.predict(np.array([1., 0., 0.]), None, scan(.3))[0] == pytest.approx(1.2)


@pytest.mark.parametrize("event", ["gap", "loss", "repeated"])
def test_restart_requires_two_fresh_matches(event):
    prior = VelocityPrior()
    pose = np.zeros(3)
    for t in (0., .1):
        prior.observe(slam.Estimate(t, pose.copy(), "matched", 1.))
    next_t = {"gap": .5, "loss": .2, "repeated": .1}[event]
    if event == "loss":
        prior.observe(slam.Estimate(.15, pose.copy(), "prediction_only", 0.))
    for t in (next_t, next_t+.1, next_t+.2):
        prior.predict(pose, None, scan(t))
        prior.observe(slam.Estimate(t, pose.copy(), "matched", 1.))
    assert prior.applied == 1


def test_synthetic_sideways_then_stop():
    rng = np.random.default_rng(88)
    fixed = rng.uniform([-3., -2.], [3., 2.], (180, 2))
    truth = []
    stream = []
    for i in range(24):
        pose = np.array([0., .04*min(i, 12), 0.])
        truth.append(pose)
        stream.append(scan(i*.1, fixed-pose[:2]))
    prior = VelocityPrior()
    baseline = list(slam.track(stream, np.zeros(3), grid=Grid(.08)))
    experimental = list(slam.track(stream, np.zeros(3), grid=Grid(.08), velocity_prior=prior))
    for rows in (baseline, experimental):
        assert [r.status for r in rows] == ["origin"] + ["matched"]*23
        errors = np.linalg.norm(np.array([r.pose[:2] for r in rows])-np.array(truth)[:, :2], axis=1)
        print("lateral mm", 1000*np.sqrt(np.mean(errors[1:13]**2)),
              "stopped mm", 1000*np.sqrt(np.mean(errors[13:]**2)))
        assert np.max(errors) < .02
    assert prior.applied == 21


@pytest.mark.parametrize("gyro,expected_yaw", [(None, .4), (.06, .06)])
def test_kinematics_take_priority_and_reset_history(gyro, expected_yaw):
    prior = VelocityPrior()
    pose = np.zeros(3)
    prior.observe(slam.Estimate(0., np.array([-.12, 0., 0.]), "matched", 1.))
    prior.observe(slam.Estimate(.1, pose.copy(), "matched", 1.))
    current = scan(.2, gyro=gyro)
    current.odom = np.array([0., .03, .4])
    prediction = prior.predict(pose, np.zeros(3), current)
    np.testing.assert_allclose(prediction, [0., .03, expected_yaw])
    assert prior.applied == 0
    assert not prior.history
