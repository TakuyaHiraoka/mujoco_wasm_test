export const EPS = 1e-8;

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function clamp01(value) {
  return clamp(value, 0, 1);
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

export function lengthSq(v) {
  return dot(v, v);
}

export function length(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

export function normalize(v) {
  const len = length(v);
  if (len < EPS) return [0, 0, 0];
  return scale(v, 1 / len);
}

export function distance(a, b) {
  return length(sub(a, b));
}

export function midpoint(a, b) {
  return scale(add(a, b), 0.5);
}

export function averagePoint(points) {
  if (!points.length) return [0, 0, 0];
  const sum = points.reduce((acc, point) => add(acc, point), [0, 0, 0]);
  return scale(sum, 1 / points.length);
}

export function orthonormalBasis(tangent) {
  const t = normalize(tangent);
  const helper = Math.abs(t[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  const u = normalize(cross(helper, t));
  const v = normalize(cross(t, u));
  return { t, u, v };
}

export function polylineLength(points) {
  let total = 0;
  for (let i = 0; i < points.length - 1; i += 1) {
    total += distance(points[i], points[i + 1]);
  }
  return total;
}

export function pointsToSegments(points, closed = false) {
  const segments = [];
  const limit = closed ? points.length : points.length - 1;
  for (let i = 0; i < limit; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    const delta = sub(b, a);
    const segLength = length(delta);
    if (segLength < EPS) continue;
    segments.push({
      index: i,
      a,
      b,
      dir: scale(delta, 1 / segLength),
      length: segLength,
    });
  }
  return segments;
}

export function formatVec(v) {
  return `${v[0].toFixed(6)} ${v[1].toFixed(6)} ${v[2].toFixed(6)}`;
}

export function formatQuat(q) {
  const nq = quatNormalize(q);
  return `${nq[0].toFixed(6)} ${nq[1].toFixed(6)} ${nq[2].toFixed(6)} ${nq[3].toFixed(6)}`;
}

export function radiansToDegrees(radians) {
  return radians * 180 / Math.PI;
}

export function quatIdentity() {
  return [1, 0, 0, 0];
}

export function quatNormalize(q) {
  const len = Math.hypot(q[0], q[1], q[2], q[3]);
  if (len < EPS) return [1, 0, 0, 0];
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

export function quatConjugate(q) {
  return [q[0], -q[1], -q[2], -q[3]];
}

export function quatMultiply(a, b) {
  return quatNormalize([
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ]);
}

export function quatFromAxisAngle(axis, angle) {
  const unit = normalize(axis);
  const half = angle * 0.5;
  const s = Math.sin(half);
  return quatNormalize([Math.cos(half), unit[0] * s, unit[1] * s, unit[2] * s]);
}

export function quatFromEulerXYZ(x, y, z) {
  const qx = quatFromAxisAngle([1, 0, 0], x);
  const qy = quatFromAxisAngle([0, 1, 0], y);
  const qz = quatFromAxisAngle([0, 0, 1], z);
  return quatMultiply(qz, quatMultiply(qy, qx));
}

export function applyQuat(q, v) {
  const nq = quatNormalize(q);
  const p = [0, v[0], v[1], v[2]];
  const qp = quatMultiply(nq, p);
  const result = quatMultiply(qp, quatConjugate(nq));
  return [result[1], result[2], result[3]];
}

export function transformPoint(point, pos, quat) {
  return add(applyQuat(quat, point), pos);
}

export function transformDirection(direction, quat) {
  return normalize(applyQuat(quat, direction));
}

export function transformSegments(segments, pos, quat) {
  return segments.map((segment) => ({
    ...segment,
    a: transformPoint(segment.a, pos, quat),
    b: transformPoint(segment.b, pos, quat),
  }));
}

export function projectPointOntoPlane(point, planeCenter, planeNormal) {
  const rel = sub(point, planeCenter);
  const signedDistance = dot(rel, planeNormal);
  return sub(point, scale(planeNormal, signedDistance));
}

export function segmentSegmentDistance(a0, a1, b0, b1) {
  const u = sub(a1, a0);
  const v = sub(b1, b0);
  const w = sub(a0, b0);
  const a = dot(u, u);
  const b = dot(u, v);
  const c = dot(v, v);
  const d = dot(u, w);
  const e = dot(v, w);
  const D = a * c - b * b;
  let sN = D;
  let tN = D;
  let sD = D;
  let tD = D;

  if (D < EPS) {
    sN = 0;
    sD = 1;
    tN = e;
    tD = c;
  } else {
    sN = b * e - c * d;
    tN = a * e - b * d;
    if (sN < 0) {
      sN = 0;
      tN = e;
      tD = c;
    } else if (sN > sD) {
      sN = sD;
      tN = e + b;
      tD = c;
    }
  }

  if (tN < 0) {
    tN = 0;
    if (-d < 0) {
      sN = 0;
    } else if (-d > a) {
      sN = sD;
    } else {
      sN = -d;
      sD = a;
    }
  } else if (tN > tD) {
    tN = tD;
    if (-d + b < 0) {
      sN = 0;
    } else if (-d + b > a) {
      sN = sD;
    } else {
      sN = -d + b;
      sD = a;
    }
  }

  const sc = Math.abs(sN) < EPS ? 0 : sN / sD;
  const tc = Math.abs(tN) < EPS ? 0 : tN / tD;
  const dP = add(w, sub(scale(u, sc), scale(v, tc)));
  return length(dP);
}

export function minSegmentDistance(segmentsA, segmentsB, skip = null) {
  let best = Infinity;
  let pair = null;
  for (let i = 0; i < segmentsA.length; i += 1) {
    for (let j = 0; j < segmentsB.length; j += 1) {
      if (skip && skip(i, j, segmentsA[i], segmentsB[j])) continue;
      const dist = segmentSegmentDistance(
        segmentsA[i].a,
        segmentsA[i].b,
        segmentsB[j].a,
        segmentsB[j].b,
      );
      if (dist < best) {
        best = dist;
        pair = [i, j];
      }
    }
  }
  return { distance: best, pair };
}
