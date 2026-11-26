"""Map-based localization core shared by the ROS 2 node and the offline replay.

State: the correction T_map_odom. For every lidar sweep:

1. prediction  T_map_base = T_map_odom * T_odom_base(stamp)   (kinematic odometry);
2. scan-to-map point-to-plane ICP from the prediction, with the prediction as a
   prior (maximum a posteriori pose given odometry and the scan);
3. quality gates; a rejected match keeps the prediction (``odometry_only``);
4. the accepted pose updates T_map_odom, projected to x, y, yaw when ``planar``.

Quality is judged on *structure* points (more than ``structure_height`` above
the floor): about half of every sweep is floor near the robot, and the floor
matches at any wrong x, y, yaw. A pose whose structure points stop matching for
``lost_after`` sweeps is declared lost and the global search runs again.

Initialization: ``odom`` (map frame = odom frame, valid when both bags share
the robot's odom frame, see docs), an explicit pose, or a global search over
places the mapping run visited.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from . import cloud, se3
from .icp import prior_information, register, residuals_to_map

MATCHED = 'matched'
ODOMETRY_ONLY = 'odometry_only'
LOST = 'lost'
INITIALIZING = 'initializing'

PLANAR_LOCK = 1e8


@dataclass
class LocalizerParams:
    reg_voxel: float = 0.10
    min_range: float = 0.3
    max_range: float = 12.0
    max_points: int = 1500            # cap on points after downsampling (keeps timing bounded)
    prior_sigma_rot: float = 0.01     # rad, uncertainty of the odometry prediction per sweep
    prior_sigma_trans: float = 0.02   # m
    planar: bool = True
    floor_height: float = -0.31       # floor z in base_link (robot standing height), m
    structure_height: float = 0.15    # points this far above the floor count as structure
    structure_gate: float = 0.10      # structure point matches if closer than this to the map, m
    min_structure_points: int = 15    # fewer: the sweep cannot confirm the pose, keep prediction
    min_structure_ratio: float = 0.5  # accept a match only above this
    min_inlier_ratio: float = 0.35
    max_correction_m: float = 0.5
    max_correction_rad: float = 0.35
    max_iterations: int = 15
    corr_schedule: tuple = (0.8, 0.5, 0.3)
    lost_after: int = 40              # consecutive rejected sweeps (~3 s) before declaring the pose lost
    # local search around the prediction (first recovery step)
    local_xy_m: float = 1.0
    local_step_m: float = 0.5
    local_yaw_deg: float = 20.0
    local_yaw_step_deg: float = 10.0
    local_min_structure: float = 0.6
    local_min_margin: float = 0.1     # over the best distinct candidate
    local_min_gain: float = 0.15      # over the score at the prediction, for jumps beyond local_free_jump_m
    local_free_jump_m: float = 0.3
    global_after: int = 10            # failed local searches before a global search
    global_interval: int = 20         # sweeps between global search attempts while lost (it takes seconds)
    probation: int = 10               # sweeps after a global relocalization that must all confirm it
    # global search over the places of the mapping run
    search_step_m: float = 0.75
    search_yaw_bins: int = 12
    search_min_structure: float = 0.65
    search_min_margin: float = 0.1
    search_iterations: int = 10


@dataclass
class StepResult:
    stamp: float
    status: str
    t_map_base: np.ndarray
    t_map_odom: np.ndarray
    inlier_ratio: float = float('nan')
    structure_ratio: float = float('nan')
    rmse: float = float('nan')
    n_points: int = 0
    n_structure: int = 0
    iterations: int = 0
    degenerate_dims: int = 0
    correction_m: float = 0.0
    correction_rad: float = 0.0
    search_score: float = float('nan')
    timings_ms: dict = field(default_factory=dict)


class Localizer:
    def __init__(self, target, params=None, places=None):
        """``target``: icp.PointMap; ``places``: optional list of 4x4 base poses of the mapping run for search."""
        self.target = target
        self.p = params or LocalizerParams()
        self.places = places
        self.t_map_odom = None
        self.fails = 0
        self.local_failures = 0
        self.unverified = False       # pose set from outside (odom/pose init) not yet confirmed by a sweep
        self.probation_left = 0       # sweeps left to confirm a global relocalization
        self.previous_good = None     # T_map_odom before a global jump, restored if probation fails
        self.since_global = 10 ** 9   # sweeps since the last global search attempt
        self.status = INITIALIZING
        self._prior = prior_information(self.p.prior_sigma_rot, self.p.prior_sigma_trans)
        if self.p.planar:
            self._prior = np.where(se3.planar_prior_mask(), PLANAR_LOCK, self._prior)

    # ----------------------------------------------------------------- init
    @property
    def initialized(self):
        return self.t_map_odom is not None

    def set_map_odom(self, t_map_odom, verified=False):
        self.t_map_odom = se3.planar(t_map_odom) if self.p.planar else t_map_odom.copy()
        self.fails = 0
        self.local_failures = 0
        self.unverified = not verified
        self.status = MATCHED

    def set_pose(self, t_map_base, t_odom_base):
        """Initialize from a base pose in the map (e.g. RViz 2D Pose Estimate)."""
        self.set_map_odom(t_map_base @ se3.inv(t_odom_base))

    def prepare(self, pts_base):
        m = cloud.valid_mask(pts_base, self.p.min_range, self.p.max_range)
        src = cloud.voxel_downsample(pts_base[m], self.p.reg_voxel)
        if len(src) > self.p.max_points:
            idx = np.linspace(0, len(src) - 1, self.p.max_points).astype(int)
            src = src[idx]
        return src

    def structure(self, src):
        return src[src[:, 2] > self.p.floor_height + self.p.structure_height]

    def structure_ratio(self, struct_pts, t_map_base):
        if len(struct_pts) == 0:
            return float('nan')
        r = residuals_to_map(se3.apply(t_map_base, struct_pts), self.target, gate=self.p.structure_gate)
        return float(np.mean(r < self.p.structure_gate))

    def _score_candidates(self, src, struct, inits, iterations, schedule):
        lock = np.where(se3.planar_prior_mask(), PLANAR_LOCK, 0.0)
        scored = []
        for init in inits:
            res = register(src, self.target, init, max_iterations=iterations, corr_schedule=schedule, prior=lock)
            scored.append((self.structure_ratio(struct, res.transform), res.transform))
        scored.sort(key=lambda s: -s[0])
        return scored

    @staticmethod
    def _best_and_second(scored, min_dt, min_dr):
        best, best_t = scored[0]
        second = 0.0
        for score, t in scored[1:]:
            dt, dr = se3.delta_norm(se3.inv(best_t) @ t)
            if dt > min_dt or dr > min_dr:
                second = score
                break
        return best, best_t, second

    def local_search(self, src, t_odom_base):
        """Search x, y, yaw around the current prediction. Returns (t_map_odom or None, best, second)."""
        struct = self.structure(src)
        if len(struct) < self.p.min_structure_points or self.t_map_odom is None:
            return None, float('nan'), float('nan')
        pred = self.t_map_odom @ t_odom_base
        p_pred = se3.planar(pred)
        steps = np.arange(-self.p.local_xy_m, self.p.local_xy_m + 1e-9, self.p.local_step_m)
        yaws = np.radians(np.arange(-self.p.local_yaw_deg, self.p.local_yaw_deg + 1e-9, self.p.local_yaw_step_deg))
        inits = []
        for dx in steps:
            for dy in steps:
                for dyaw in yaws:
                    shift = se3.make(se3.rot_z(dyaw), (dx, dy, 0.0))
                    # rotate about the robot, then shift in the map frame
                    move = se3.make(None, p_pred[:3, 3]) @ shift @ se3.make(None, -p_pred[:3, 3])
                    inits.append(move @ pred)
        scored = self._score_candidates(src, struct, inits, 6, (0.6, 0.3))
        best, best_t, second = self._best_and_second(scored, 0.4, np.radians(8))
        at_pred = self._score_candidates(src, struct, [pred], 6, (0.6, 0.3))[0][0]
        jump, _ = se3.delta_norm(se3.inv(pred) @ best_t)
        clear = best - second >= self.p.local_min_margin and best - at_pred >= self.p.local_min_gain
        if best >= self.p.local_min_structure and (jump <= self.p.local_free_jump_m or clear):
            return best_t @ se3.inv(t_odom_base), best, second
        return None, best, second

    def search(self, src, t_odom_base):
        """Global search over mapped places and yaw.

        Returns (t_map_odom or None, best score, second-best score of a distinct pose).
        """
        if self.places is None or len(self.places) == 0:
            return None, float('nan'), float('nan')
        struct = self.structure(src)
        if len(struct) < self.p.min_structure_points:
            return None, float('nan'), float('nan')
        xy = np.array([p[:2, 3] for p in self.places])
        keys = np.unique(np.floor(xy / self.p.search_step_m).astype(int), axis=0)
        cand_xy = (keys + 0.5) * self.p.search_step_m
        p_odom = se3.planar(t_odom_base)
        inits = []
        for x, y in cand_xy:
            for k in range(self.p.search_yaw_bins):
                yaw = 2 * np.pi * k / self.p.search_yaw_bins
                inits.append(se3.make(se3.rot_z(yaw), (x, y, 0.0)) @ se3.inv(p_odom) @ t_odom_base)
        scored = self._score_candidates(src, struct, inits, self.p.search_iterations, (1.0, 0.6, 0.3))
        best, best_t, second = self._best_and_second(scored, 1.0, 0.5)
        if best >= self.p.search_min_structure and best - second >= self.p.search_min_margin:
            return best_t @ se3.inv(t_odom_base), best, second
        return None, best, second

    # ----------------------------------------------------------------- step
    def step(self, stamp, pts_base, t_odom_base):
        """Process one deskewed sweep (points in base_link at ``stamp``)."""
        tim = {}
        t0 = time.perf_counter()
        src = self.prepare(pts_base)
        tim['prepare'] = (time.perf_counter() - t0) * 1e3

        search_score = float('nan')
        if not self.initialized or self.status == LOST:
            t1 = time.perf_counter()
            corr, kind = None, None
            if self.initialized:
                corr, search_score, _ = self.local_search(src, t_odom_base)
                kind = 'local'
                if corr is None:
                    self.local_failures += 1
            self.since_global += 1
            # an unconfirmed start goes global after one failed local search, a tracked pose waits longer
            wait = 1 if self.unverified else self.p.global_after
            if corr is None and (not self.initialized or (self.local_failures >= wait
                                                          and self.since_global >= self.p.global_interval)):
                self.since_global = 0
                corr, search_score, _ = self.search(src, t_odom_base)
                kind = 'global'
                if corr is not None and self.initialized:
                    self.previous_good = self.t_map_odom.copy()
                    self.probation_left = self.p.probation
            tim['search'] = (time.perf_counter() - t1) * 1e3
            if corr is None:
                tim['total'] = (time.perf_counter() - t0) * 1e3
                keep = self.t_map_odom if self.t_map_odom is not None else np.eye(4)
                status = LOST if self.initialized else INITIALIZING
                return StepResult(stamp, status, keep @ t_odom_base, keep, n_points=len(src),
                                  search_score=search_score, timings_ms=tim)
            failures = self.local_failures if kind == 'local' else 0
            self.set_map_odom(corr, verified=True)
            self.local_failures = failures if kind == 'local' else 0

        pred = self.t_map_odom @ t_odom_base
        t1 = time.perf_counter()
        res = register(src, self.target, pred, max_iterations=self.p.max_iterations,
                       corr_schedule=self.p.corr_schedule, prior=self._prior)
        tim['icp'] = (time.perf_counter() - t1) * 1e3
        struct = self.structure(src)
        sratio = self.structure_ratio(struct, res.transform)
        dt, dr = se3.delta_norm(se3.inv(pred) @ res.transform)
        enough_structure = len(struct) >= self.p.min_structure_points
        ok = (enough_structure and sratio >= self.p.min_structure_ratio
              and res.inlier_ratio >= self.p.min_inlier_ratio
              and dt <= self.p.max_correction_m and dr <= self.p.max_correction_rad)
        if ok:
            new = res.transform @ se3.inv(t_odom_base)
            self.t_map_odom = se3.planar(new) if self.p.planar else new
            self.fails = 0
            self.local_failures = 0
            self.unverified = False
            self.status = MATCHED
            if self.probation_left:
                self.probation_left -= 1
                if not self.probation_left:
                    self.previous_good = None
        else:
            if enough_structure:  # a sweep without structure says nothing about the pose
                self.fails += 1
            if enough_structure and self.probation_left and self.previous_good is not None:
                # the global relocalization is not confirmed: go back to the pose we had
                self.t_map_odom = self.previous_good
                self.previous_good = None
                self.probation_left = 0
                self.fails = self.p.lost_after
            if enough_structure and self.unverified:
                # an externally given start that the first sweep contradicts: search right away
                self.fails = self.p.lost_after
            self.status = LOST if self.fails >= self.p.lost_after else ODOMETRY_ONLY
        t_map_base = self.t_map_odom @ t_odom_base
        tim['total'] = (time.perf_counter() - t0) * 1e3
        return StepResult(stamp, self.status, t_map_base, self.t_map_odom.copy(), res.inlier_ratio, sratio,
                          res.rmse, len(src), len(struct), res.iterations, res.degenerate_dims, dt, dr,
                          search_score, tim)
