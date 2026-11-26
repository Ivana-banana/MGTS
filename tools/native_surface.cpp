// Optional DLL: per-scan core of the Go2 3D tracker in C++17 (#190).
// Mirrors mgts_nav.localize3d.SurfaceMap.align and .score: point-to-plane ICP with
// Cauchy weights and observable-axis updates. The normals and planar flags are computed
// once in Python and passed in.
//
// Nearest neighbours: points bucketed into cubic cells of MAX_MATCH_M; a query scans
// the 27 cells around it and skips cells that cannot hold a closer point. Within the
// radius the result equals cKDTree; beyond it the tracker only needs "miss".
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define MGTS_EXPORT extern "C" __declspec(dllexport)
#else
#define MGTS_EXPORT extern "C"
#endif

namespace {
constexpr std::int64_t kBias = 1 << 20;
constexpr int kIterations = 6;
constexpr double kCauchy = 0.08;
constexpr double kObservable = 0.008;
constexpr double kMaxShift = 0.60;
constexpr double kMaxTurn = 0.35;
constexpr double kScoreSlack = 0.01;
constexpr double kInf = std::numeric_limits<double>::infinity();

std::uint64_t pack(std::int64_t x, std::int64_t y, std::int64_t z) noexcept {
    return (static_cast<std::uint64_t>(x + kBias) << 42) |
           (static_cast<std::uint64_t>(y + kBias) << 21) | static_cast<std::uint64_t>(z + kBias);
}

bool in_range(std::int64_t v) noexcept { return v > -kBias && v < kBias; }

struct Index {
    double cell = 0, radius2 = 0;
    std::vector<double> xyz;           // reordered by cell
    std::vector<std::int64_t> source;  // original index of each reordered point
    std::unordered_map<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>> ranges;

    std::int64_t coord(double v) const noexcept { return static_cast<std::int64_t>(std::floor(v / cell)); }

    bool build(const double* points, std::size_t n, double radius) {
        cell = radius;
        radius2 = radius * radius;
        std::vector<std::uint64_t> keys(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double* p = points + 3 * i;
            const std::int64_t c[3] = {coord(p[0]), coord(p[1]), coord(p[2])};
            if (!in_range(c[0]) || !in_range(c[1]) || !in_range(c[2])) return false;
            keys[i] = pack(c[0], c[1], c[2]);
        }
        std::vector<std::uint32_t> order(n);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return keys[a] < keys[b]; });
        xyz.resize(3 * n);
        source.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto i = order[k];
            std::copy(points + 3 * i, points + 3 * i + 3, xyz.begin() + 3 * k);
            source[k] = i;
            ranges.try_emplace(keys[i], static_cast<std::uint32_t>(k), 0u).first->second.second += 1;
        }
        return true;
    }

    // Original index of the nearest point closer than the radius, or -1.
    std::int64_t nearest(const double* p, double& distance) const {
        const std::int64_t c[3] = {coord(p[0]), coord(p[1]), coord(p[2])};
        double best = radius2;
        std::int64_t found = -1;
        for (int dx = -1; dx <= 1; ++dx) {
            const double lx = (c[0] + dx) * cell, gx = std::max({lx - p[0], 0.0, p[0] - lx - cell});
            for (int dy = -1; dy <= 1; ++dy) {
                const double ly = (c[1] + dy) * cell, gy = std::max({ly - p[1], 0.0, p[1] - ly - cell});
                for (int dz = -1; dz <= 1; ++dz) {
                    const double lz = (c[2] + dz) * cell, gz = std::max({lz - p[2], 0.0, p[2] - lz - cell});
                    if (gx * gx + gy * gy + gz * gz >= best) continue;
                    const std::int64_t cx = c[0] + dx, cy = c[1] + dy, cz = c[2] + dz;
                    if (!in_range(cx) || !in_range(cy) || !in_range(cz)) continue;
                    const auto it = ranges.find(pack(cx, cy, cz));
                    if (it == ranges.end()) continue;
                    const double* q = xyz.data() + 3 * it->second.first;
                    for (std::uint32_t k = 0; k < it->second.second; ++k, q += 3) {
                        const double ex = q[0] - p[0], ey = q[1] - p[1], ez = q[2] - p[2];
                        const double d2 = ex * ex + ey * ey + ez * ez;
                        if (d2 < best) { best = d2; found = it->second.first + k; }
                    }
                }
            }
        }
        if (found < 0) { distance = kInf; return -1; }
        distance = std::sqrt(best);
        return source[found];
    }
};

struct Quat { double x, y, z, w; };

Quat mul(const Quat& a, const Quat& b) {
    Quat q{a.w * b.x + b.w * a.x + (a.y * b.z - a.z * b.y),
           a.w * b.y + b.w * a.y + (a.z * b.x - a.x * b.z),
           a.w * b.z + b.w * a.z + (a.x * b.y - a.y * b.x),
           a.w * b.w - (a.x * b.x + a.y * b.y + a.z * b.z)};
    const double s = 1.0 / std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return {q.x * s, q.y * s, q.z * s, q.w * s};
}

Quat from_rotvec(const double* v) {
    const double angle = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const double a2 = angle * angle;
    // Same small-angle series as scipy.
    const double scale = angle <= 1e-3 ? 0.5 - a2 / 48 + a2 * a2 / 3840 : std::sin(angle / 2) / angle;
    return {v[0] * scale, v[1] * scale, v[2] * scale, std::cos(angle / 2)};
}

struct Pose {
    Quat q;
    double t[3];
    void matrix(double r[9]) const {
        const double x = q.x, y = q.y, z = q.z, w = q.w;
        r[0] = 1 - 2 * (y * y + z * z); r[1] = 2 * (x * y - z * w); r[2] = 2 * (x * z + y * w);
        r[3] = 2 * (x * y + z * w); r[4] = 1 - 2 * (x * x + z * z); r[5] = 2 * (y * z - x * w);
        r[6] = 2 * (x * z - y * w); r[7] = 2 * (y * z + x * w); r[8] = 1 - 2 * (x * x + y * y);
    }
};

void rotate(const double r[9], const double* p, double* out) {
    out[0] = r[0] * p[0] + r[1] * p[1] + r[2] * p[2];
    out[1] = r[3] * p[0] + r[4] * p[1] + r[5] * p[2];
    out[2] = r[6] * p[0] + r[7] * p[1] + r[8] * p[2];
}

// Cyclic Jacobi for a symmetric 6x6 matrix: eigenvalues in `values`, vectors in columns of `v`.
void eigh6(double a[6][6], double values[6], double v[6][6]) {
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) v[i][j] = i == j;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0, total = 0;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) (i == j ? total : off) += a[i][j] * a[i][j];
        if (off <= 1e-30 * (total + off)) break;
        for (int p = 0; p < 5; ++p) {
            for (int q = p + 1; q < 6; ++q) {
                if (a[p][q] == 0) continue;
                const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                const double t = (theta >= 0 ? 1 : -1) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < 6; ++k) {
                    const double akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 6; ++k) {
                    const double apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 6; ++k) {
                    const double vkp = v[k][p], vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (int i = 0; i < 6; ++i) values[i] = a[i][i];
}

struct Surface {
    Index full, planes;
    std::vector<double> normals, plane_normals;
    std::vector<std::uint8_t> planar;
    double radius = 0;
    // Scratch buffers reused across scans.
    std::vector<double> rotated, norms;

    double score(const double* points, std::size_t n, const Pose& pose) const {
        if (!n) return std::numeric_limits<double>::quiet_NaN();
        double r[9], world[3], d, sum = 0;
        pose.matrix(r);
        for (std::size_t i = 0; i < n; ++i) {
            rotate(r, points + 3 * i, world);
            for (int k = 0; k < 3; ++k) world[k] += pose.t[k];
            full.nearest(world, d);
            sum += std::min(d, radius);
        }
        return sum / static_cast<double>(n);
    }

    // 0: too few matches (Python None); 1: `out`/`accepted` hold the result.
    int fit(const double* points, std::size_t n, const Pose& initial, const Index& tree,
            const double* tree_normals, const std::uint8_t* valid, Pose& out, bool& accepted) {
        Pose pose = initial;
        rotated.resize(3 * n);
        norms.resize(n);
        std::vector<std::int64_t> match(n);
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            double r[9], world[3], d;
            pose.matrix(r);
            std::size_t count = 0;
            for (std::size_t i = 0; i < n; ++i) {
                double* rot = rotated.data() + 3 * i;
                rotate(r, points + 3 * i, rot);
                for (int k = 0; k < 3; ++k) world[k] = rot[k] + pose.t[k];
                std::int64_t j = tree.nearest(world, d);
                if (j >= 0 && valid && !valid[j]) j = -1;
                match[i] = j;
                if (j >= 0) norms[count++] = std::sqrt(rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2]);
            }
            if (static_cast<double>(count) < std::max(35.0, 0.25 * static_cast<double>(n))) return 0;
            const std::size_t mid = count / 2;
            std::nth_element(norms.begin(), norms.begin() + mid, norms.begin() + count);
            double median = norms[mid];
            if (count % 2 == 0) median = (median + *std::max_element(norms.begin(), norms.begin() + mid)) / 2;
            const double scale = std::max(1.0, median);

            double h[6][6] = {}, g[6] = {};
            for (std::size_t i = 0; i < n; ++i) {
                const std::int64_t j = match[i];
                if (j < 0) continue;
                const double* rot = rotated.data() + 3 * i;
                const double* nm = tree_normals + 3 * j;
                const double* target = tree_points + 3 * j;
                double error = 0;
                for (int k = 0; k < 3; ++k) error += (rot[k] + pose.t[k] - target[k]) * nm[k];
                const double jac[6] = {nm[0], nm[1], nm[2],
                                       (rot[1] * nm[2] - rot[2] * nm[1]) / scale,
                                       (rot[2] * nm[0] - rot[0] * nm[2]) / scale,
                                       (rot[0] * nm[1] - rot[1] * nm[0]) / scale};
                const double ratio = error / kCauchy;
                const double w2 = 1.0 / (1.0 + ratio * ratio);
                for (int a = 0; a < 6; ++a) {
                    g[a] += jac[a] * w2 * error;
                    for (int b = a; b < 6; ++b) h[a][b] += jac[a] * jac[b] * w2;
                }
            }
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < a; ++b) h[a][b] = h[b][a];
            double values[6], vectors[6][6];
            eigh6(h, values, vectors);
            const double largest = *std::max_element(values, values + 6);
            double delta[6] = {};
            for (int axis = 0; axis < 6; ++axis) {
                if (values[axis] < kObservable * largest || values[axis] <= 0) continue;
                double along = 0;
                for (int k = 0; k < 6; ++k) along += vectors[k][axis] * g[k];
                for (int k = 0; k < 6; ++k) delta[k] -= vectors[k][axis] * along / values[axis];
            }
            const double turn[3] = {delta[3] / scale, delta[4] / scale, delta[5] / scale};
            pose.q = mul(from_rotvec(turn), pose.q);
            for (int k = 0; k < 3; ++k) pose.t[k] += delta[k];
            double size = 0;
            for (double v : delta) size += v * v;
            if (std::sqrt(size) < 0.001) break;
        }
        const double sx = pose.t[0] - initial.t[0], sy = pose.t[1] - initial.t[1], sz = pose.t[2] - initial.t[2];
        const Quat inv{-initial.q.x, -initial.q.y, -initial.q.z, initial.q.w};
        const Quat rel = mul(pose.q, inv);
        const double turn = 2 * std::atan2(std::sqrt(rel.x * rel.x + rel.y * rel.y + rel.z * rel.z), std::abs(rel.w));
        if (std::sqrt(sx * sx + sy * sy + sz * sz) > kMaxShift || turn > kMaxTurn ||
            score(points, n, pose) > score(points, n, initial) + kScoreSlack) {
            out = initial;
            accepted = false;
        } else {
            out = pose;
            accepted = true;
        }
        return 1;
    }

    const double* tree_points = nullptr;  // original-order points of the tree being fitted
    std::vector<double> full_points, plane_points;
};

Pose read_pose(const double* v) { return {{v[0], v[1], v[2], v[3]}, {v[4], v[5], v[6]}}; }
void write_pose(const Pose& p, double* v) {
    v[0] = p.q.x; v[1] = p.q.y; v[2] = p.q.z; v[3] = p.q.w;
    v[4] = p.t[0]; v[5] = p.t[1]; v[6] = p.t[2];
}
}  // namespace

// points/normals: n x 3; planar: n flags; plane points/normals: m x 3 (rows where planar).
MGTS_EXPORT void* surf_create(const double* points, const double* normals, const std::uint8_t* planar,
                              std::size_t n, const double* plane_points, const double* plane_normals,
                              std::size_t m, double radius) {
    if (!(radius > 0) || n >= std::numeric_limits<std::uint32_t>::max()) return nullptr;
    auto* s = new (std::nothrow) Surface;
    if (!s) return nullptr;
    try {
        s->radius = radius;
        s->full_points.assign(points, points + 3 * n);
        s->plane_points.assign(plane_points, plane_points + 3 * m);
        s->normals.assign(normals, normals + 3 * n);
        s->plane_normals.assign(plane_normals, plane_normals + 3 * m);
        s->planar.assign(planar, planar + n);
        if (!s->full.build(points, n, radius) || !s->planes.build(plane_points, m, radius)) {
            delete s;
            return nullptr;
        }
    } catch (...) {
        delete s;
        return nullptr;
    }
    return s;
}

MGTS_EXPORT void surf_destroy(void* handle) { delete static_cast<Surface*>(handle); }

// pose arrays: qx qy qz qw x y z. Returns 0 on success.
MGTS_EXPORT int surf_align(void* handle, const double* points, std::size_t n, const double* initial,
                           double* out, int* accepted) {
    auto* s = static_cast<Surface*>(handle);
    if (!s) return 1;
    try {
        const Pose start = read_pose(initial);
        Pose result = start;
        bool ok = false;
        s->tree_points = s->full_points.data();
        int found = s->fit(points, n, start, s->full, s->normals.data(), s->planar.data(), result, ok);
        if (!found && !s->plane_points.empty()) {
            // An edge or isolated return can hide a usable plane at the same distance scale.
            s->tree_points = s->plane_points.data();
            found = s->fit(points, n, start, s->planes, s->plane_normals.data(), nullptr, result, ok);
        }
        if (!found) { result = start; ok = false; }
        write_pose(result, out);
        *accepted = ok;
    } catch (...) {
        return 2;
    }
    return 0;
}

MGTS_EXPORT double surf_score(void* handle, const double* points, std::size_t n, const double* pose) {
    const auto* s = static_cast<const Surface*>(handle);
    return s ? s->score(points, n, read_pose(pose)) : std::numeric_limits<double>::quiet_NaN();
}
