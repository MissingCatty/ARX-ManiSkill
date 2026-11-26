// The same convex-solid test as geometry.py, batched across source candidates.
// CPU only; preserve facet order, stage order and strict penetration margins.
#include <cstdint>
#include <vector>

extern "C" int arx_cavity_counts(
    std::int64_t candidates, const double* poses, std::int64_t points_count,
    const double* points, const double* lower, const double* upper,
    std::int64_t* counts) {
  for (std::int64_t candidate = 0; candidate < candidates; ++candidate) {
    const double* pose = poses + 16 * candidate;
    std::int64_t hits = 0;
    for (std::int64_t point = 0; point < points_count; ++point) {
      const double* p = points + 3 * point;
      const double x = p[0] - pose[3];
      const double y = p[1] - pose[7];
      const double z = p[2] - pose[11];
      bool inside = true;
      for (int axis = 0; axis < 3; ++axis) {
        const double value = (x * pose[axis] + y * pose[4 + axis]) + z * pose[8 + axis];
        if (value < lower[axis] || value > upper[axis]) {
          inside = false;
          break;
        }
      }
      hits += inside;
    }
    counts[candidate] = hits;
  }
  return 0;
}

extern "C" int arx_point_rejections(
    std::int64_t candidates, const double* poses, const double* points,
    const std::int64_t* point_offsets, const std::int64_t* point_ids,
    std::int64_t parts, const double* bounds, const double* planes,
    const std::int64_t* plane_offsets, double margin, double pre_distance,
    std::uint8_t* rejected, std::int64_t* counts) {
  const double shifts[3] = {-pre_distance, -pre_distance / 2., 0.};
  std::vector<double> local;
  for (std::int64_t candidate = 0; candidate < candidates; ++candidate) {
    const double* pose = poses + 16 * candidate;
    const auto first = point_offsets[candidate];
    const auto size = point_offsets[candidate + 1] - first;
    local.resize(3 * size);
    for (std::int64_t i = 0; i < size; ++i) {
      const double* p = points + 3 * point_ids[first + i];
      const double x = p[0] - pose[3];
      const double y = p[1] - pose[7];
      const double z = p[2] - pose[11];
      for (int axis = 0; axis < 3; ++axis)
        local[3 * i + axis] = (x * pose[axis] + y * pose[4 + axis]) + z * pose[8 + axis];
    }
    rejected[candidate] = 0;
    counts[candidate] = 0;
    for (double shift : shifts) {
      for (std::int64_t part = 0; part < parts; ++part) {
        const auto plane_first = plane_offsets[part];
        const auto plane_end = plane_offsets[part + 1];
        if (plane_first == plane_end) continue;  // Planar pieces have no solid.
        const double* b = bounds + 6 * part;
        std::int64_t hits = 0;
        for (std::int64_t i = 0; i < size; ++i) {
          const double x = local[3 * i];
          const double y = local[3 * i + 1];
          const double z = local[3 * i + 2] - shift;
          if (x < b[0] + margin || x > b[3] - margin ||
              y < b[1] + margin || y > b[4] - margin ||
              z < b[2] + margin || z > b[5] - margin) continue;
          bool inside = true;
          for (auto face = plane_first; face < plane_end; ++face) {
            const double* f = planes + 4 * face;
            if (((x * f[0] + y * f[1]) + z * f[2]) + f[3] > -margin) {
              inside = false;
              break;
            }
          }
          hits += inside;
        }
        if (hits) {
          counts[candidate] = hits;
          rejected[candidate] = 1;
          break;
        }
      }
      if (rejected[candidate]) break;
    }
  }
  return 0;
}
