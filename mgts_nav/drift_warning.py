"Online scan consistency checks used by the frozen drift-warning experiment."
import numpy as np
from .observability import diagnose
from .slam import relative, world

def predict(scan, previous, pose, previous_pose, support, tree, odom_mode):
    try:
        geometry = diagnose(scan.xy)['normal_eigenvalues'][0]
    except ValueError:
        geometry = None
    distances, _ = tree.query(world(scan.xy, pose))
    disagreement = False
    if previous is not None and 0 < scan.t - previous.t <= .25:
        delta = relative(previous_pose, pose)
        if scan.gyro_delta is not None:
            angle = delta[2] - scan.gyro_delta
            disagreement = abs(np.arctan2(np.sin(angle), np.cos(angle))) > np.deg2rad(3)
        if odom_mode != 'absent' and previous.odom is not None and scan.odom is not None:
            odom = relative(previous.odom, scan.odom)
            if odom_mode == 'drift':
                odom[0] += .08 * (scan.t - previous.t)
            disagreement |= np.linalg.norm(delta[:2] - odom[:2]) > .05
    votes = [support < .55, np.quantile(distances, .75) > .25,
             geometry is not None and geometry < .12, disagreement]
    return bool(votes[0]), sum(map(int, votes)) >= 2


