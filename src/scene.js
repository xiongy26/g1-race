// three.js 场景: 程序化赛道 + 基于 mjv 渲染管线的机器人可视化。
// 世界坐标系与 MuJoCo 一致: z 轴向上。
//
// 机器人渲染走 mjv_updateScene 管线(与官方 demo / g1-boxing-wasm 相同):
// 每个 geom 的最终世界位姿由 MuJoCo 计算好(pos + 3x3 mat), 自动包含
// mesh 顶点坐标系对齐, 避免手工组合 body×geom×mesh 变换导致的"散开"。
// 注意: 本 @mujoco/mujoco WASM 构建中 mjvGeom.dataid 对 mesh geom 会返回
// 损坏值(真实值的 2 倍), 需要用 geom_dataid[objid] 还原真实 mesh id。

import * as THREE from 'three';

export const FINISH_X = 25;      // 终点线
export const LANE_WIDTH = 1.35;  // 道宽
export const TRACK_W = 11;       // 赛道总宽(含缓冲)
export const TRACK_X0 = -4;      // 赛道纹理覆盖范围
export const TRACK_X1 = 38;

export const TEAM_COLORS = ['#ff4d4d', '#ffb400', '#37c871', '#3a9bff', '#c85bff', '#ff6fb0'];

export const MAX_LANES = 6;
export function laneY(i) { return (i - (MAX_LANES - 1) / 2) * LANE_WIDTH; }

// ---------- 赛道 ----------
export function buildTrack(scene) {
  const w = 2048, h = 560;
  const s = w / (TRACK_X1 - TRACK_X0); // px per meter
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const g = cv.getContext('2d');
  const X = (xm) => (xm - TRACK_X0) * s;
  const Y = (ym) => (TRACK_W / 2 - ym) * s; // y 翻转使 +y 向上

  const halfW = (MAX_LANES * LANE_WIDTH) / 2;
  // 外围
  g.fillStyle = '#8a9299';
  g.fillRect(0, 0, w, h);
  // 跑道
  g.fillStyle = '#b2513c';
  g.fillRect(0, Y(halfW), w, (halfW * 2) * s);

  // 5 米刻度与标号
  g.font = `${Math.round(0.5 * s)}px sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle';
  for (let xm = 0; xm <= 30; xm += 5) {
    const px = X(xm);
    g.fillStyle = 'rgba(255,255,255,0.85)';
    g.fillRect(px - 1.5, Y(halfW), 3, halfW * 2 * s);
    g.fillStyle = 'rgba(255,255,255,0.9)';
    g.fillText(`${xm}m`, px, Y(halfW + 0.55));
  }

  // 道线
  g.fillStyle = '#ffffff';
  for (let i = 0; i <= MAX_LANES; i++) {
    const y = Y(-halfW + i * LANE_WIDTH);
    g.fillRect(0, y - 2, w, 4);
  }

  // 道号
  for (let i = 0; i < MAX_LANES; i++) {
    const ym = -halfW + (i + 0.5) * LANE_WIDTH;
    g.save();
    g.translate(X(-1.5), Y(ym));
    g.fillStyle = 'rgba(255,255,255,0.92)';
    g.font = `bold ${Math.round(0.8 * s)}px sans-serif`;
    g.fillText(String(i + 1), 0, 0);
    g.restore();
  }

  // 起点线(绿)
  g.fillStyle = '#2ea44f';
  g.fillRect(X(0) - 0.2 * s, Y(halfW), 0.4 * s, halfW * 2 * s);

  // 终点线(方格旗)
  const fx = X(FINISH_X);
  const cols = 12, rows = 4;
  const cw = (0.7 * s) / cols, ch = (halfW * 2 * s) / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      g.fillStyle = (r + c) % 2 === 0 ? '#111' : '#fff';
      g.fillRect(fx - 0.35 * s + c * cw, Y(halfW) + r * ch, cw, ch);
    }
  }
  g.fillStyle = '#111';
  g.font = `bold ${Math.round(0.6 * s)}px sans-serif`;
  g.fillText('FINISH 25m', fx, Y(halfW + 0.75));

  const tex = new THREE.CanvasTexture(cv);
  tex.anisotropy = 8;
  tex.colorSpace = THREE.SRGBColorSpace;

  const geo = new THREE.PlaneGeometry(TRACK_X1 - TRACK_X0, TRACK_W);
  const mat = new THREE.MeshPhongMaterial({ map: tex, shininess: 4 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set((TRACK_X0 + TRACK_X1) / 2, 0, 0.005);
  mesh.receiveShadow = true;
  scene.add(mesh);

  // 赛道外围大地
  const groundGeo = new THREE.PlaneGeometry(400, 400);
  const groundMat = new THREE.MeshPhongMaterial({ color: 0x6fa35f, shininess: 0 });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.position.set(15, 0, -0.02);
  scene.add(ground);

  // 终点拱门
  const arch = new THREE.Group();
  const pillarMat = new THREE.MeshPhongMaterial({ color: 0xf3f5f7 });
  for (const y of [-halfW - 0.3, halfW + 0.3]) {
    const pillar = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 2.6), pillarMat);
    pillar.position.set(FINISH_X, y, 1.3);
    pillar.castShadow = true;
    arch.add(pillar);
  }
  const bannerCv = document.createElement('canvas');
  bannerCv.width = 512; bannerCv.height = 96;
  const bg = bannerCv.getContext('2d');
  bg.fillStyle = '#1b2733'; bg.fillRect(0, 0, 512, 96);
  bg.fillStyle = '#ffb000'; bg.font = 'bold 56px sans-serif';
  bg.textAlign = 'center'; bg.textBaseline = 'middle';
  bg.fillText('FINISH · 25m', 256, 50);
  const bannerTex = new THREE.CanvasTexture(bannerCv);
  const banner = new THREE.Mesh(
    new THREE.PlaneGeometry(2 * halfW + 0.6, 0.7),
    new THREE.MeshBasicMaterial({ map: bannerTex, side: THREE.DoubleSide })
  );
  banner.position.set(FINISH_X, 0, 2.35);
  arch.add(banner);
  scene.add(arch);
}

// ---------- mjv 共享状态 ----------
let sharedOption = null, sharedPerturb = null, sharedCamera = null;
const meshGeoCache = new Map(); // meshId -> BufferGeometry(全机器人共享)
const primGeoCache = new Map(); // 图元 key -> BufferGeometry

function ensureShared(mj) {
  if (sharedOption) return;
  sharedOption = new mj.MjvOption();
  for (let i = 0; i < 6; i++) {
    sharedOption.geomgroup[i] = 0;
    sharedOption.sitegroup[i] = 0;
  }
  sharedOption.geomgroup[1] = 1; // 本模型: group 1 = 可视网格; group 0 = 碰撞网格(隐藏)
  sharedPerturb = new mj.MjvPerturb();
  sharedCamera = new mj.MjvCamera();
}

function getMeshGeometry(model, meshId) {
  let g = meshGeoCache.get(meshId);
  if (g) return g;
  const va = model.mesh_vertadr[meshId], vn = model.mesh_vertnum[meshId];
  const fa = model.mesh_faceadr[meshId], fn = model.mesh_facenum[meshId];
  const positions = new Float32Array(vn * 3);
  for (let i = 0; i < vn * 3; i++) positions[i] = model.mesh_vert[3 * va + i];
  const indices = new Uint32Array(fn * 3);
  for (let i = 0; i < fn * 3; i++) indices[i] = model.mesh_face[3 * fa + i];
  g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeVertexNormals();
  meshGeoCache.set(meshId, g);
  return g;
}

function getPrimGeometry(mj, type, size) {
  const key = type + ':' + size.join(',');
  let g = primGeoCache.get(key);
  if (g) return g;
  const G = mj.mjtGeom;
  if (type === G.mjGEOM_SPHERE.value) g = new THREE.SphereGeometry(size[0], 24, 16);
  else if (type === G.mjGEOM_CAPSULE.value) {
    g = new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 16);
    g.rotateX(Math.PI / 2); // mujoco capsule 沿 z
  } else if (type === G.mjGEOM_CYLINDER.value) {
    g = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 28);
    g.rotateX(Math.PI / 2);
  } else if (type === G.mjGEOM_BOX.value) {
    g = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
  } else if (type === G.mjGEOM_PLANE.value) {
    g = new THREE.PlaneGeometry(size[0] ? 2 * size[0] : 12, size[1] ? 2 * size[1] : 12);
  } else {
    g = new THREE.SphereGeometry(0.02, 8, 6);
  }
  primGeoCache.set(key, g);
  return g;
}

// ---------- 机器人 ----------
export class RobotVisual {
  constructor(mj, model, teamColor, label) {
    this.mj = mj;
    this.model = model;
    ensureShared(mj);
    this.mjvScene = new mj.MjvScene(model, 20000);
    this.group = new THREE.Group();
    this.meshes = []; // mjv geom 槽位 -> three mesh
    this.tint = new THREE.Color(teamColor);

    // 头顶编号牌
    const cv = document.createElement('canvas');
    cv.width = 128; cv.height = 128;
    const c = cv.getContext('2d');
    c.beginPath(); c.arc(64, 64, 56, 0, Math.PI * 2);
    c.fillStyle = teamColor; c.fill();
    c.lineWidth = 6; c.strokeStyle = '#fff'; c.stroke();
    c.fillStyle = '#fff'; c.font = 'bold 72px sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(String(label), 64, 70);
    const tex = new THREE.CanvasTexture(cv);
    this.sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: true }));
    this.sprite.scale.set(0.42, 0.42, 1);
    this.group.add(this.sprite);
  }

  // 由 MjData 更新位姿
  update(data) {
    const mj = this.mj;
    const model = this.model;
    mj.mjv_updateScene(model, data, sharedOption, sharedPerturb, sharedCamera,
      mj.mjtCatBit.mjCAT_ALL.value, this.mjvScene);
    const geoms = this.mjvScene.geoms;
    const n = geoms.size();
    try {
      for (let i = 0; i < n; i++) {
        const gm = geoms.get(i);
        const type = Number(gm.type);
        const objtype = Number(gm.objtype), objid = Number(gm.objid);
        if (objtype !== mj.mjtObj.mjOBJ_GEOM.value) { gm.delete(); continue; }
        const size = Array.from(gm.size).map(Number);
        const rgba = Array.from(gm.rgba);
        const mat9 = Array.from(gm.mat);
        const pos = Array.from(gm.pos);
        gm.delete();

        // 修复该 WASM 构建的 dataid 损坏问题: 用 objid 查真实 mesh id
        let meshId = -1;
        if (type === mj.mjtGeom.mjGEOM_MESH.value) {
          meshId = (objid >= 0 && objid < model.ngeom) ? Number(model.geom_dataid[objid]) : Number.NaN;
          if (!Number.isFinite(meshId)) continue;
        }
        const geo = type === mj.mjtGeom.mjGEOM_MESH.value
          ? getMeshGeometry(model, meshId)
          : getPrimGeometry(mj, type, size);
        const key = type === mj.mjtGeom.mjGEOM_MESH.value ? `m${meshId}` : `p${type}:${size.join(',')}`;

        let mesh = this.meshes[i];
        if (!mesh || mesh.userData.key !== key) {
          if (mesh) this.group.remove(mesh);
          mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ shininess: 24, specular: 0x222222 }));
          mesh.castShadow = true;
          mesh.userData.key = key;
          this.meshes[i] = mesh;
          this.group.add(mesh);
        }

        // 白色/浅色部件染队伍色, 深色部件保持原色
        const lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2];
        if (lum > 0.25) {
          mesh.material.color.setRGB(
            rgba[0] * 0.55 + this.tint.r * 0.45,
            rgba[1] * 0.55 + this.tint.g * 0.45,
            rgba[2] * 0.55 + this.tint.b * 0.45);
        } else {
          mesh.material.color.setRGB(rgba[0], rgba[1], rgba[2]);
        }

        mesh.matrixAutoUpdate = false;
        mesh.matrix.set(
          mat9[0], mat9[1], mat9[2], pos[0],
          mat9[3], mat9[4], mat9[5], pos[1],
          mat9[6], mat9[7], mat9[8], pos[2],
          0, 0, 0, 1);
        mesh.matrixWorldNeedsUpdate = true;
        mesh.visible = true;
      }
    } finally {
      geoms.delete();
    }
    for (let i = n; i < this.meshes.length; i++) {
      if (this.meshes[i]) this.meshes[i].visible = false;
    }
    // 编号牌跟随 pelvis(body 1)
    const xpos = data.xpos;
    this.sprite.position.set(xpos[3], xpos[4], xpos[5] + 1.25);
  }

  dispose() {
    try { this.mjvScene.delete(); } catch (e) { /* 已释放 */ }
    this.group.traverse((o) => { if (o.material) o.material.dispose(); });
  }
}

// ---------- 灯光 ----------
export function addLights(scene) {
  scene.add(new THREE.HemisphereLight(0xdfeaff, 0x88a06a, 0.85));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(-10, -8, 20);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 80;
  sun.shadow.camera.left = -16;
  sun.shadow.camera.right = 16;
  sun.shadow.camera.top = 16;
  sun.shadow.camera.bottom = -16;
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  scene.add(sun.target);
  return sun;
}
