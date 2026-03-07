export const EPS = 1e-8;

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function vec3(x = 0, y = 0, z = 0) {
  return [x, y, z];
}

export function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function length(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

export function lengthSq(v) {
  return dot(v, v);
}

export function normalize(v) {
  const len = length(v);
  if (len < EPS) {
    return [0, 0, 0];
  }
  return scale(v, 1 / len);
}

export function distance(a, b) {
  return length(sub(a, b));
}

export function midpoint(a, b) {
  return scale(add(a, b), 0.5);
}

export function almostEqualVec(a, b, eps = EPS) {
  return distance(a, b) < eps;
}

export function orthonormalBasis(tangent) {
  const t = normalize(tangent);
  const helper = Math.abs(t[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  const u = normalize(cross(helper, t));
  const v = normalize(cross(t, u));
  return { t, u, v };
}

export function rotateBasisAroundAxis(u, v, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const u2 = add(scale(u, c), scale(v, s));
  const v2 = add(scale(v, c), scale(u, -s));
  return { u: u2, v: v2 };
}

export function formatVec(v) {
  return `${v[0].toFixed(6)} ${v[1].toFixed(6)} ${v[2].toFixed(6)}`;
}

export function formatQuat(q) {
  return `${q[0].toFixed(6)} ${q[1].toFixed(6)} ${q[2].toFixed(6)} ${q[3].toFixed(6)}`;
}

export function quatNormalize(q) {
  const len = Math.hypot(q[0], q[1], q[2], q[3]);
  if (len < EPS) {
    return [1, 0, 0, 0];
  }
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

export function quatFromUnitVectors(a, b) {
  const v1 = normalize(a);
  const v2 = normalize(b);
  const r = dot(v1, v2) + 1;

  if (r < EPS) {
    const axis = Math.abs(v1[0]) > Math.abs(v1[2])
      ? normalize([-v1[1], v1[0], 0])
      : normalize([0, -v1[2], v1[1]]);
    return quatNormalize([0, axis[0], axis[1], axis[2]]);
  }

  const c = cross(v1, v2);
  return quatNormalize([r, c[0], c[1], c[2]]);
}

export function polylineLength(points) {
  let total = 0;
  for (let i = 0; i < points.length - 1; i += 1) {
    total += distance(points[i], points[i + 1]);
  }
  return total;
}

export function polylineProgress(points, point) {
  let bestDistanceSq = Infinity;
  let accumulated = 0;
  let bestAlong = 0;

  for (let i = 0; i < points.length - 1; i += 1) {
    const a = points[i];
    const b = points[i + 1];
    const ab = sub(b, a);
    const abLenSq = lengthSq(ab);
    if (abLenSq < EPS) continue;

    const ap = sub(point, a);
    const t = clamp(dot(ap, ab) / abLenSq, 0, 1);
    const proj = add(a, scale(ab, t));
    const dSq = lengthSq(sub(point, proj));
    if (dSq < bestDistanceSq) {
      bestDistanceSq = dSq;
      bestAlong = accumulated + Math.sqrt(abLenSq) * t;
    }
    accumulated += Math.sqrt(abLenSq);
  }

  return bestAlong;
}
