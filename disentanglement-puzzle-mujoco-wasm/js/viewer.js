import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { computePoseGeometry } from './diagnostics.js';

class CapsuleGeometry extends THREE.BufferGeometry {
  constructor(radius = 1, length = 1, capSegments = 8, radialSegments = 16) {
    const path = new THREE.Path();
    path.absarc(0, -length / 2, radius, Math.PI * 1.5, 0, false);
    path.absarc(0, length / 2, radius, 0, Math.PI * 0.5, false);

    const lathe = new THREE.LatheGeometry(path.getPoints(capSegments), radialSegments);

    super();
    this.setIndex(lathe.getIndex());
    this.setAttribute('position', lathe.getAttribute('position'));
    this.setAttribute('normal', lathe.getAttribute('normal'));
    this.setAttribute('uv', lathe.getAttribute('uv'));
    this.type = 'CapsuleGeometry';
  }
}

function setLinePositions(line, a, b) {
  const array = line.geometry.attributes.position.array;
  array[0] = a.x;
  array[1] = a.y;
  array[2] = a.z;
  array[3] = b.x;
  array[4] = b.y;
  array[5] = b.z;
  line.geometry.attributes.position.needsUpdate = true;
  line.geometry.computeBoundingSphere();
}

export class MujocoThreeBridge {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05070e);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute('aria-label', '3D viewer');
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-1.9, -3.4, 1.9);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, 0);

    this.simGroup = new THREE.Group();
    this.debugGroup = new THREE.Group();
    this.scene.add(this.simGroup);
    this.scene.add(this.debugGroup);

    this.meshes = [];
    this.geometryCache = new Map();
    this.debugObjects = null;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.58));

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.18);
    keyLight.position.set(2.3, -2.8, 4.8);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    this.scene.add(keyLight);

    const rimLight = new THREE.PointLight(0x69d7ff, 1.0, 16, 2);
    rimLight.position.set(-2.4, 2.2, 2.5);
    this.scene.add(rimLight);

    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  getGeometry(mujoco, mjvGeom) {
    const key = JSON.stringify([
      mjvGeom.type,
      Array.from(mjvGeom.size || []),
      mjvGeom.dataid,
    ]);

    if (this.geometryCache.has(key)) {
      return this.geometryCache.get(key);
    }

    let geometry;

    if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
      geometry = new THREE.PlaneGeometry(
        2 * (mjvGeom.size[0] || 1000),
        2 * (mjvGeom.size[1] || 1000),
      );
      const uv = geometry.getAttribute('uv');
      for (let i = 0; i < uv.count; i += 1) {
        uv.setY(i, 1 - uv.getY(i));
      }
    } else if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(mjvGeom.size[0], 24, 16);
    } else if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
      geometry = new CapsuleGeometry(mjvGeom.size[0], 2 * mjvGeom.size[2], 12, 16);
      geometry.rotateX(0.5 * Math.PI);
    } else if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(
        2 * mjvGeom.size[0],
        2 * mjvGeom.size[1],
        2 * mjvGeom.size[2],
      );
    } else if (mjvGeom.type === mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(
        mjvGeom.size[0],
        mjvGeom.size[0],
        2 * mjvGeom.size[2],
        24,
      );
      geometry.rotateX(0.5 * Math.PI);
    } else {
      geometry = new THREE.SphereGeometry(0.01, 8, 8);
    }

    this.geometryCache.set(key, geometry);
    return geometry;
  }

  clearSimulationMeshes() {
    this.meshes.forEach((mesh) => {
      if (mesh.parent) mesh.parent.remove(mesh);
      if (mesh.material) mesh.material.dispose();
    });
    this.meshes.length = 0;

    this.geometryCache.forEach((geometry) => geometry.dispose());
    this.geometryCache.clear();
  }

  clearDebug() {
    while (this.debugGroup.children.length > 0) {
      const child = this.debugGroup.children[0];
      this.debugGroup.remove(child);

      child.traverse?.((node) => {
        if (node.geometry) node.geometry.dispose();
        if (node.material) {
          if (Array.isArray(node.material)) {
            node.material.forEach((material) => material.dispose());
          } else {
            node.material.dispose();
          }
        }
      });
    }

    this.debugObjects = null;
  }

  setDebugVisible(visible) {
    this.debugGroup.visible = visible;
  }

  setDebugMarkers(spec) {
    this.clearDebug();

    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));

    const fixedCenter = new THREE.Mesh(
      new THREE.SphereGeometry(spec.wire.radius * 0.55, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xf5f5f5 }),
    );
    fixedCenter.position.set(0, 0, 0);

    const movingCenter = new THREE.Mesh(
      new THREE.SphereGeometry(spec.wire.radius * 0.55, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0x3be4ff }),
    );

    const fixedGapLine = new THREE.Line(
      lineGeometry.clone(),
      new THREE.LineBasicMaterial({ color: 0xffb74d }),
    );

    const movingGapLine = new THREE.Line(
      lineGeometry.clone(),
      new THREE.LineBasicMaterial({ color: 0xff4fc9 }),
    );

    const centerLinkLine = new THREE.Line(
      lineGeometry.clone(),
      new THREE.LineDashedMaterial({
        color: 0x8ff7b5,
        dashSize: 0.08,
        gapSize: 0.05,
        opacity: 0.72,
        transparent: true,
      }),
    );

    const fixedArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0, 0),
      spec.fixed.primaryLoopRadius * 0.34,
      0x7ef6a8,
      spec.wire.radius * 2.6,
      spec.wire.radius * 1.6,
    );

    const movingArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 0),
      spec.moving.primaryLoopRadius * 0.34,
      0x80d8ff,
      spec.wire.radius * 2.6,
      spec.wire.radius * 1.6,
    );

    setLinePositions(
      fixedGapLine,
      new THREE.Vector3(...spec.fixed.gap.aLocal),
      new THREE.Vector3(...spec.fixed.gap.bLocal),
    );

    fixedArrow.position.set(0, 0, 0);
    fixedArrow.setDirection(new THREE.Vector3(0, 0, 1));

    this.debugGroup.add(
      fixedCenter,
      movingCenter,
      fixedGapLine,
      movingGapLine,
      centerLinkLine,
      fixedArrow,
      movingArrow,
    );

    this.debugObjects = {
      fixedCenter,
      movingCenter,
      fixedGapLine,
      movingGapLine,
      centerLinkLine,
      fixedArrow,
      movingArrow,
    };

    this.updateMovingMarkers(spec, {
      pos: spec.startPose.pos,
      quat: spec.startPose.quat,
    });
  }

  updateMovingMarkers(spec, pose) {
    if (!this.debugObjects) return;

    const markers = computePoseGeometry(spec, pose);
    const fixedCenter = new THREE.Vector3(...markers.fixed.holeCenter);
    const movingCenter = new THREE.Vector3(...markers.moving.holeCenter);

    this.debugObjects.fixedCenter.position.copy(fixedCenter);
    this.debugObjects.movingCenter.position.copy(movingCenter);

    setLinePositions(this.debugObjects.centerLinkLine, fixedCenter, movingCenter);
    this.debugObjects.centerLinkLine.computeLineDistances();

    setLinePositions(
      this.debugObjects.movingGapLine,
      new THREE.Vector3(...markers.moving.gapA),
      new THREE.Vector3(...markers.moving.gapB),
    );

    this.debugObjects.movingArrow.position.copy(movingCenter);
    this.debugObjects.movingArrow.setDirection(
      new THREE.Vector3(...markers.moving.holeNormal).normalize(),
    );
    this.debugObjects.movingArrow.setLength(
      spec.moving.primaryLoopRadius * 0.34,
      spec.wire.radius * 2.6,
      spec.wire.radius * 1.6,
    );
  }

  sync(mujoco, model, data, mjvScene, mjvOption, mjvPerturb, mjvCamera) {
    mujoco.mjv_updateScene(
      model,
      data,
      mjvOption,
      mjvPerturb,
      mjvCamera,
      mujoco.mjtCatBit.mjCAT_ALL.value,
      mjvScene,
    );

    const geoms = mjvScene.geoms;
    const count = geoms.size();

    for (let i = 0; i < count; i += 1) {
      const mjvGeom = geoms.get(i);
      if (!mjvGeom) continue;

      const geometry = this.getGeometry(mujoco, mjvGeom);
      let mesh = this.meshes[i];

      if (!mesh || mesh.userData.geometry !== geometry) {
        if (mesh) {
          this.simGroup.remove(mesh);
          if (mesh.material) mesh.material.dispose();
        }

        const material = new THREE.MeshPhongMaterial({
          color: new THREE.Color(mjvGeom.rgba[0], mjvGeom.rgba[1], mjvGeom.rgba[2]),
          transparent: mjvGeom.rgba[3] < 0.999,
          opacity: mjvGeom.rgba[3],
          side:
            mjvGeom.type === mujoco.mjtGeom.mjGEOM_PLANE.value
              ? THREE.DoubleSide
              : THREE.FrontSide,
        });

        mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData.geometry = geometry;
        this.meshes[i] = mesh;
        this.simGroup.add(mesh);
      }

      mesh.material.color.setRGB(mjvGeom.rgba[0], mjvGeom.rgba[1], mjvGeom.rgba[2]);
      mesh.material.opacity = mjvGeom.rgba[3];
      mesh.material.transparent = mjvGeom.rgba[3] < 0.999;
      mesh.matrixAutoUpdate = false;
      mesh.matrix.set(
        mjvGeom.mat[0], mjvGeom.mat[1], mjvGeom.mat[2], mjvGeom.pos[0],
        mjvGeom.mat[3], mjvGeom.mat[4], mjvGeom.mat[5], mjvGeom.pos[1],
        mjvGeom.mat[6], mjvGeom.mat[7], mjvGeom.mat[8], mjvGeom.pos[2],
        0, 0, 0, 1,
      );
      mesh.matrixWorldNeedsUpdate = true;

      mjvGeom.delete();
    }

    for (let i = count; i < this.meshes.length; i += 1) {
      const mesh = this.meshes[i];
      if (!mesh) continue;
      if (mesh.parent) mesh.parent.remove(mesh);
      if (mesh.material) mesh.material.dispose();
    }

    this.meshes.length = count;
    geoms.delete();
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.clearSimulationMeshes();
    this.clearDebug();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
