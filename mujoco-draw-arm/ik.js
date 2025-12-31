// 3リンク(平面)アームの簡易IK（解析解）
// 目標: 末端(ペン先)の (x, y) と姿勢 phi を満たす joint angles [q1, q2, q3]
// ここでは、(x, y) は MuJoCo world 座標系の XY 平面上の点を想定します。

export function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

export function wrapToPi(rad) {
  // (-pi, pi] に折り返し
  const twoPi = Math.PI * 2;
  let r = rad % twoPi;
  if (r <= -Math.PI) r += twoPi;
  if (r > Math.PI) r -= twoPi;
  return r;
}

/**
 * 3リンク平面アームのIK（wrist中心を 2リンクIK で解く）
 * @param {number} x  末端目標 x (world)
 * @param {number} y  末端目標 y (world)
 * @param {number} L1 リンク1長
 * @param {number} L2 リンク2長
 * @param {number} L3 リンク3長（末端オフセット）
 * @param {number} phi 末端姿勢（平面内角度, world XY）
 * @param {"down"|"up"} elbow エルボー解の選択
 * @returns {[number, number, number] | null}
 */
export function ik3LinkPlanar(x, y, L1, L2, L3, phi, elbow = "down") {
  // wrist 位置 = 末端 - L3 * [cos(phi), sin(phi)]
  const wx = x - L3 * Math.cos(phi);
  const wy = y - L3 * Math.sin(phi);

  const r2 = wx * wx + wy * wy;
  const c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2);

  if (c2 < -1.0 - 1e-9 || c2 > 1.0 + 1e-9) {
    return null; // 到達不能
  }

  const c2c = clamp(c2, -1.0, 1.0);
  let q2 = Math.acos(c2c);
  if (elbow === "up") q2 = -q2;

  const k1 = L1 + L2 * Math.cos(q2);
  const k2 = L2 * Math.sin(q2);

  const q1 = Math.atan2(wy, wx) - Math.atan2(k2, k1);
  const q3 = phi - q1 - q2;

  return [wrapToPi(q1), wrapToPi(q2), wrapToPi(q3)];
}
