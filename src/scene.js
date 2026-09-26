// three.js 场景: 程序化赛道 + 基于 mjv 渲染管线的机器人可视化。
// 世界坐标系与 MuJoCo 一致: z 轴向上。
//
// 机器人渲染走 mjv_updateScene 管线(与官方 demo / g1-boxing-wasm 相同):
// 每个 geom 的最终世界位姿由 MuJoCo 计算好(pos + 3x3 mat), 自动包含
// mesh 顶点坐标系对齐, 避免手工组合 body×geom×mesh 变换导致的"散开"。
// 注意: 本 @mujoco/mujoco WASM 构建中 mjvGeom.dataid 对 mesh geom 会返回
// 损坏值(真实值的 2 倍), 需要用 geom_dataid[objid] 还原真实 mesh id。

import * as THREE from 'three';
import { FENCE_INNER_Y, FENCE_CENTER_X, FENCE_HALF_LEN, FENCE_HALF_T, makeRng } from './sim.js';

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

  // 赛道外围大地(割草纹理草坪)
  const groundGeo = new THREE.PlaneGeometry(400, 400);
  const groundMat = new THREE.MeshPhongMaterial({ map: makeGrassTexture(), shininess: 0 });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.position.set(15, 0, -0.02);
  ground.receiveShadow = true;
  scene.add(ground);

  // 赛道两侧红白路缘(与 MuJoCo 里的 race_fence 物理挡墙一一对应)
  const kerbCv = document.createElement('canvas');
  kerbCv.width = 256; kerbCv.height = 32;
  const kg = kerbCv.getContext('2d');
  for (let i = 0; i < 16; i++) {
    kg.fillStyle = i % 2 === 0 ? '#d43a2f' : '#f5f0e6';
    kg.fillRect(i * 16, 0, 16, 32);
  }
  const kerbTex = new THREE.CanvasTexture(kerbCv);
  kerbTex.colorSpace = THREE.SRGBColorSpace;
  kerbTex.wrapS = THREE.RepeatWrapping;
  kerbTex.repeat.set(FENCE_HALF_LEN * 2 / 2, 1);
  const kerbGeo = new THREE.BoxGeometry(FENCE_HALF_LEN * 2, FENCE_HALF_T * 2 + 0.02, 0.22);
  const kerbMat = new THREE.MeshPhongMaterial({ map: kerbTex, shininess: 6 });
  for (const side of [-1, 1]) {
    const kerb = new THREE.Mesh(kerbGeo, kerbMat);
    kerb.position.set(FENCE_CENTER_X, side * (FENCE_INNER_Y + FENCE_HALF_T), 0.11);
    kerb.receiveShadow = true;
    scene.add(kerb);
  }

  // 终点拱门(与起点拱门同构, 横幅面向 -x 侧镜头)
  makeGate(scene, FINISH_X, 'FINISH · 25m', '#ffb000');
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

function getMeshGeometry(model, meshId, cacheKey = '') {
  const key = cacheKey + meshId;
  let g = meshGeoCache.get(key);
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
  meshGeoCache.set(key, g);
  return g;
}

function getPrimGeometry(mj, type, size) {
  const key = type + ':' + size.join(',');
  let g = primGeoCache.get(key);
  if (g) return g;
  const G = mj.mjtGeom;
  if (type === G.mjGEOM_SPHERE.value) g = new THREE.SphereGeometry(size[0], 24, 16);
  else if (type === G.mjGEOM_ELLIPSOID.value) {
    g = new THREE.SphereGeometry(1, 24, 16);
    g.scale(size[0], size[1], size[2]);
  } else if (type === G.mjGEOM_CAPSULE.value) {
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
  constructor(mj, model, teamColor, label, labelH = 1.25, cacheKey = '', visGroups = [1]) {
    this.mj = mj;
    this.model = model;
    ensureShared(mj);
    this.mjvScene = new mj.MjvScene(model, 20000);
    // 每物种可见 geom 组(官方 MJCF 的视觉网格组各不相同: G1/T1=1, PM01=2)
    this.option = new mj.MjvOption();
    for (let i = 0; i < 6; i++) this.option.geomgroup[i] = 0;
    for (let i = 0; i < 6; i++) this.option.sitegroup[i] = 0;
    for (const g of visGroups) this.option.geomgroup[g] = 1;
    this.group = new THREE.Group();
    this.meshes = []; // mjv geom 槽位 -> three mesh
    this.tint = new THREE.Color(teamColor);
    this.labelH = labelH;
    this.cacheKey = cacheKey;

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
    mj.mjv_updateScene(model, data, this.option, sharedPerturb, sharedCamera,
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
        const isMesh = type === mj.mjtGeom.mjGEOM_MESH.value;
        const geo = isMesh
          ? getMeshGeometry(model, meshId, this.cacheKey)
          : getPrimGeometry(mj, type, size);
        const key = isMesh ? `m${this.cacheKey}${meshId}` : `p${type}:${size.join(',')}`;

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
    // 编号牌跟随 pelvis(body 1), 高度随物种
    const xpos = data.xpos;
    this.sprite.position.set(xpos[3], xpos[4], xpos[5] + this.labelH);
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

// ---------- 环境景观(跑道以外) ----------
// 渐变天空穹顶/太阳/漂动云、远山剪影、树木、两侧看台+彩色观众、起点拱门、
// 广告围挡、终点直道彩旗与气球束。静态元素合批/实例化, 每帧只更新云与气球,
// 不进物理、不挡跑道(|y|>=9.5 留白带), 会被录制画布一并收录。
const scenery = { clouds: [], balloons: [] };

export function buildEnvironment(scene) {
  const rng = makeRng(0xbeef01);
  buildSky(scene);
  buildSunAndClouds(scene, rng);
  buildHills(scene);
  buildTrees(scene, rng);
  buildStand(scene, { x0: 5, width: 16, y0: 8.2, dir: 1, steps: 4, fascia: '🏃 双足短跑大赛 · BIPEDAL SPRINT GP' });
  buildStand(scene, { x0: -2, width: 13, y0: -8.6, dir: -1, steps: 3, fascia: '' });
  makeGate(scene, 0, 'START · 0m', '#38d97a');
  buildBoards(scene);
  buildBunting(scene);
  buildBalloons(scene);
}

// 每帧调用: 云漂移 + 气球浮动
export function updateScenery(dt, t) {
  for (const c of scenery.clouds) {
    c.sp.position.x += c.speed * dt;
    if (c.sp.position.x > 140) c.sp.position.x = -140;
  }
  for (const b of scenery.balloons) {
    b.grp.position.z = Math.sin(t * 1.2 + b.phase) * 0.12;
    b.grp.rotation.z = Math.sin(t * 0.9 + b.phase) * 0.04;
  }
}

function buildSky(scene) {
  const geo = new THREE.SphereGeometry(320, 32, 16);
  geo.rotateX(Math.PI / 2); // 球极轴转到 z(世界向上)
  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  const top = new THREE.Color(0x3d7fc7), horizon = new THREE.Color(0xcfe8f7);
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const h = Math.max(0, Math.min(1, pos.getZ(i) / 320 / 0.55));
    c.copy(horizon).lerp(top, h * h * (3 - 2 * h));
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const dome = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    vertexColors: true, side: THREE.BackSide, fog: false, depthWrite: false,
  }));
  dome.renderOrder = -10;
  scene.add(dome);
}

function buildSunAndClouds(scene, rng) {
  const sun = new THREE.Sprite(new THREE.SpriteMaterial({
    map: makeGlowTexture(), fog: false, depthWrite: false, opacity: 0.95,
  }));
  sun.position.set(125, -75, 115);
  sun.scale.set(30, 30, 1);
  scene.add(sun);
  for (let i = 0; i < 9; i++) {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({
      map: makeCloudTexture(0xc100 + i * 97), fog: false, depthWrite: false,
      opacity: 0.55 + rng() * 0.3, rotation: (rng() - 0.5) * 0.2,
    }));
    sp.position.set(-120 + rng() * 240, (rng() < 0.5 ? -1 : 1) * (30 + rng() * 90), 46 + rng() * 26);
    const s = 16 + rng() * 20;
    sp.scale.set(s, s * 0.45, 1);
    scenery.clouds.push({ sp, speed: 0.5 + rng() * 1.0 });
    scene.add(sp);
  }
}

// 远山剪影: 沿跑道两侧与终点远端的竖直条带, 双面 + 雾化成层次
function buildHills(scene) {
  const layers = [
    { axis: 'x', pos: [0, -70, 0], u0: -180, u1: 210, color: 0x7fa06a, amp: [7, 4], f: [0.045, 0.11], ph: [0.7, 2.1] },
    { axis: 'x', pos: [0, 78, 0], u0: -180, u1: 215, color: 0x76985f, amp: [9, 5], f: [0.04, 0.09], ph: [1.9, 4.2] },
    { axis: 'x', pos: [0, -150, 0], u0: -220, u1: 260, color: 0x9db6c6, amp: [16, 9], f: [0.03, 0.07], ph: [0.2, 3.3] },
    { axis: 'x', pos: [0, 158, 0], u0: -220, u1: 260, color: 0x9db6c6, amp: [14, 10], f: [0.032, 0.08], ph: [2.8, 5.1] },
    { axis: 'y', pos: [205, 0, 0], u0: -160, u1: 160, color: 0xa9bfd0, amp: [18, 10], f: [0.028, 0.075], ph: [1.2, 4.8] },
  ];
  for (const L of layers) {
    const hFn = (u) => 2.5
      + L.amp[0] * (0.5 + 0.5 * Math.sin(u * L.f[0] + L.ph[0]))
      + L.amp[1] * (0.5 + 0.5 * Math.sin(u * L.f[1] + L.ph[1]));
    const mesh = new THREE.Mesh(
      ridgeGeometry(L.u0, L.u1, 64, L.axis, hFn),
      new THREE.MeshBasicMaterial({ color: L.color, side: THREE.DoubleSide })
    );
    mesh.position.set(...L.pos);
    scene.add(mesh);
  }
}

function ridgeGeometry(u0, u1, segs, axis, hFn) {
  const pos = [], idx = [];
  for (let i = 0; i <= segs; i++) {
    const u = u0 + ((u1 - u0) * i) / segs;
    const h = hFn(u);
    if (axis === 'x') pos.push(u, 0, 0, u, 0, h);
    else pos.push(0, u, 0, 0, u, h);
    if (i < segs) {
      const a = i * 2;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  return g;
}

function buildTrees(scene, rng) {
  const N = 90;
  const pts = [];
  let guard = 0;
  while (pts.length < N && guard++ < 3000) {
    const x = -35 + rng() * 100;
    const y = (rng() < 0.5 ? -1 : 1) * (7.5 + rng() * 60);
    if (y > 0 && x > 3 && x < 23 && y < 17) continue;       // +y 主看台
    if (y < 0 && x > -3 && x < 14 && y > -15.5) continue;   // -y 看台
    if (Math.abs(y) < 9.5 && x > -6 && x < 30) continue;    // 跑道两侧留白带
    pts.push([x, y]);
  }
  const trunks = new THREE.InstancedMesh(
    new THREE.CylinderGeometry(0.08, 0.13, 0.9, 6),
    new THREE.MeshPhongMaterial({ color: 0x7a5b3a }), pts.length);
  const crowns = new THREE.InstancedMesh(
    new THREE.IcosahedronGeometry(0.65, 0),
    new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 4 }), pts.length);
  trunks.castShadow = crowns.castShadow = true;
  trunks.frustumCulled = crowns.frustumCulled = false;
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion();
  const p = new THREE.Vector3(), s = new THREE.Vector3(), c = new THREE.Color();
  const zAxis = new THREE.Vector3(0, 0, 1), leafA = new THREE.Color(0x4e8a3c), leafB = new THREE.Color(0x86c05e);
  pts.forEach(([x, y], k) => {
    const sc = 0.8 + rng() * 1.1;
    q.setFromAxisAngle(zAxis, rng() * Math.PI * 2);
    p.set(x, y, 0.45 * sc); s.set(sc, sc, sc);
    m4.compose(p, q, s); trunks.setMatrixAt(k, m4);
    p.set(x, y, 1.25 * sc);
    s.set(sc * (0.85 + rng() * 0.4), sc * (0.85 + rng() * 0.4), sc * (1.0 + rng() * 0.5));
    m4.compose(p, q, s); crowns.setMatrixAt(k, m4);
    crowns.setColorAt(k, c.copy(leafA).lerp(leafB, rng()));
  });
  trunks.instanceMatrix.needsUpdate = true;
  crowns.instanceMatrix.needsUpdate = true;
  if (crowns.instanceColor) crowns.instanceColor.needsUpdate = true;
  scene.add(trunks, crowns);
}

// 看台: 阶梯混凝土 + 实例化彩色观众 + 白色顶棚(+ 主看台横幅)
function buildStand(scene, { x0, width, y0, dir, steps, fascia }) {
  const grp = new THREE.Group();
  const concrete = new THREE.MeshPhongMaterial({ color: 0xcfd6dd });
  for (let i = 0; i < steps; i++) {
    const step = new THREE.Mesh(new THREE.BoxGeometry(width, 1.3, 0.5 * (i + 1)), concrete);
    step.position.set(x0 + width / 2, dir * (y0 + 1.3 * i + 0.65), 0.25 * (i + 1));
    step.receiveShadow = true;
    grp.add(step);
  }
  const rng = makeRng(0x5eed + steps * 7);
  const palette = [...TEAM_COLORS, '#f5f5f5', '#2e3a48', '#ffb000', '#7fd4ff'];
  const people = [];
  for (let i = 0; i < steps; i++) {
    for (let row = 0; row < 2; row++) {
      const py = dir * (y0 + 1.3 * i + 0.4 + row * 0.45);
      const pz = 0.5 * (i + 1) + 0.2;
      for (let px = x0 + 0.4; px < x0 + width - 0.3; px += 0.46) {
        people.push({
          p: new THREE.Vector3(px + (rng() - 0.5) * 0.1, py + (rng() - 0.5) * 0.08, pz),
          c: new THREE.Color(palette[(rng() * palette.length) | 0]),
          s: 0.85 + rng() * 0.4,
        });
      }
    }
  }
  const crowd = new THREE.InstancedMesh(new THREE.BoxGeometry(0.26, 0.2, 0.36), new THREE.MeshLambertMaterial(), people.length);
  crowd.frustumCulled = false;
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3();
  people.forEach((it, k) => {
    s.set(it.s, it.s, it.s);
    m4.compose(it.p, q, s);
    crowd.setMatrixAt(k, m4);
    crowd.setColorAt(k, it.c);
  });
  crowd.instanceMatrix.needsUpdate = true;
  if (crowd.instanceColor) crowd.instanceColor.needsUpdate = true;
  grp.add(crowd);

  const roof = new THREE.Mesh(
    new THREE.BoxGeometry(width + 1.2, 1.3 * steps + 1.0, 0.1),
    new THREE.MeshPhongMaterial({ color: 0xf4f6f8 }));
  roof.position.set(x0 + width / 2, dir * (y0 + 1.3 * steps * 0.5), 0.5 * steps + 2.0);
  roof.rotation.x = dir * 0.1; // 顶棚向跑道一侧倾斜
  grp.add(roof);
  const poleMat = new THREE.MeshPhongMaterial({ color: 0xaeb8c2 });
  for (const cx of [x0 + 0.6, x0 + width - 0.6]) {
    for (const cy of [y0 + 0.3, y0 + 1.3 * steps - 0.2]) {
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.5 * steps + 2.0, 8), poleMat);
      pole.position.set(cx, dir * cy, (0.5 * steps + 2.0) / 2);
      grp.add(pole);
    }
  }
  if (fascia) {
    const plate = makeTextPlate(width - 0.6, 0.72, '#141a22', fascia, '#ffb000');
    plate.position.set(x0 + width / 2, dir * (y0 - 0.12), 0.5 * steps + 1.55);
    plate.up.set(0, 0, 1);
    plate.lookAt(plate.position.x, plate.position.y - dir * 10, plate.position.z);
    grp.add(plate);
  }
  scene.add(grp);
}

// 赛道两侧广告围挡
function buildBoards(scene) {
  const items = [
    { x: 4, side: 1, text: 'MUJOCO 3.14 · WASM 物理' },
    { x: 11.5, side: 1, text: 'ONNX RUNTIME WEB' },
    { x: 19, side: 1, text: '官方模型 × 官方策略' },
    { x: 7.5, side: -1, text: 'THREE.JS 实时渲染' },
    { x: 15, side: -1, text: '25M · 6 LANES · RL' },
    { x: 22.5, side: -1, text: '双足短跑大赛 🏃' },
  ];
  for (const it of items) {
    const grp = new THREE.Group();
    const back = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.1, 1.05), new THREE.MeshPhongMaterial({ color: 0x232f3d }));
    back.position.z = 0.5;
    const plate = makeTextPlate(3.4, 0.85, '#141a22', it.text, '#e8eef5');
    plate.position.z = 0.58;
    grp.add(back, plate);
    grp.position.set(it.x, it.side * 6.35, 0);
    grp.up = new THREE.Vector3(0, 0, 1);
    grp.lookAt(it.x, it.side * 6.35 - it.side * 10, 0.5); // 板面朝向跑道
    scene.add(grp);
  }
}

// 终点直道两侧彩旗(旗串 + 旗杆)
function buildBunting(scene) {
  const xs = [21, 23.2, 25.4];
  const pos = [], col = [];
  const flagColors = TEAM_COLORS.map((h) => new THREE.Color(h));
  const stringMat = new THREE.LineBasicMaterial({ color: 0x666f7a });
  for (const side of [-1, 1]) {
    const y = side * 5.7;
    for (const x of xs) {
      const pole = new THREE.Mesh(
        new THREE.CylinderGeometry(0.05, 0.05, 2.6, 6),
        new THREE.MeshPhongMaterial({ color: 0x8892a0 }));
      pole.position.set(x, y, 1.3);
      scene.add(pole);
    }
    for (let i = 0; i < xs.length - 1; i++) {
      const x0 = xs[i], x1 = xs[i + 1];
      const sag = (t) => 2.45 - 0.35 * Math.sin(Math.PI * t);
      const pts = [];
      for (let k = 0; k <= 8; k++) pts.push(new THREE.Vector3(x0 + (x1 - x0) * k / 8, y, sag(k / 8)));
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), stringMat));
      for (let k = 0; k < 7; k++) {
        const t = (k + 0.5) / 7;
        const px = x0 + (x1 - x0) * t, pz = sag(t);
        pos.push(px - 0.16, y, pz, px + 0.16, y, pz, px, y, pz - 0.42);
        const c = flagColors[k % flagColors.length];
        col.push(c.r, c.g, c.b, c.r, c.g, c.b, c.r, c.g, c.b);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  const flags = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  flags.frustumCulled = false;
  scene.add(flags);
}

// 起终点气球束(随风轻摆)
function buildBalloons(scene) {
  const cols = [0xff4d4d, 0xffb400, 0x3a9bff];
  for (const [bx, by] of [[0, 5.4], [28.5, 5.4]]) {
    const grp = new THREE.Group();
    for (let i = 0; i < 3; i++) {
      const b = new THREE.Mesh(
        new THREE.SphereGeometry(0.34, 20, 14),
        new THREE.MeshPhongMaterial({ color: cols[i], shininess: 60 }));
      const ang = -0.5 + i * 0.5;
      b.position.set(Math.sin(ang) * 0.55, Math.cos(ang) * 0.35, 3.1 + i * 0.42);
      b.scale.set(1, 1, 1.25);
      grp.add(b);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([b.position.clone(), new THREE.Vector3(0, 0, 0.05)]),
        new THREE.LineBasicMaterial({ color: 0x9aa4b0 }));
      grp.add(line);
    }
    grp.position.set(bx, by, 0);
    scene.add(grp);
    scenery.balloons.push({ grp, phase: bx });
  }
}

// 起点/终点通用拱门: 双立柱 + 面向 -x 的横幅(选手与镜头自 -x 侧接近)
function makeGate(scene, x, label, accent) {
  const halfW = (MAX_LANES * LANE_WIDTH) / 2;
  const arch = new THREE.Group();
  const pillarMat = new THREE.MeshPhongMaterial({ color: 0xf3f5f7 });
  for (const y of [-halfW - 0.3, halfW + 0.3]) {
    const pillar = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 2.6), pillarMat);
    pillar.position.set(x, y, 1.3);
    pillar.castShadow = true;
    arch.add(pillar);
  }
  const cv = document.createElement('canvas');
  cv.width = 512; cv.height = 96;
  const g = cv.getContext('2d');
  g.fillStyle = '#1b2733'; g.fillRect(0, 0, 512, 96);
  g.fillStyle = accent; g.font = 'bold 52px sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(label, 256, 50);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  const banner = new THREE.Mesh(
    new THREE.PlaneGeometry(2 * halfW + 0.6, 0.7),
    new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide })
  );
  banner.position.set(x, 0, 2.35);
  banner.up.set(0, 0, 1);
  banner.lookAt(x - 1, 0, 2.35);
  arch.add(banner);
  scene.add(arch);
}

function makeTextPlate(w, h, bg, text, color) {
  const cv = document.createElement('canvas');
  cv.width = 1024;
  cv.height = Math.max(48, Math.round((1024 * h) / w));
  const g = cv.getContext('2d');
  g.fillStyle = bg;
  g.fillRect(0, 0, cv.width, cv.height);
  g.fillStyle = color;
  g.font = `bold ${Math.round(cv.height * 0.42)}px "Microsoft YaHei", sans-serif`;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText(text, cv.width / 2, cv.height / 2 + 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ map: tex })
  );
}

function makeGlowTexture() {
  const cv = document.createElement('canvas');
  cv.width = 128; cv.height = 128;
  const g = cv.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,252,240,1)');
  grad.addColorStop(0.25, 'rgba(255,246,214,0.85)');
  grad.addColorStop(1, 'rgba(255,246,214,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function makeCloudTexture(seed) {
  const cv = document.createElement('canvas');
  cv.width = 256; cv.height = 128;
  const g = cv.getContext('2d');
  const rng = makeRng(seed);
  for (let i = 0; i < 11; i++) {
    const x = 40 + rng() * 176, y = 45 + rng() * 40, r = 22 + rng() * 30;
    const grad = g.createRadialGradient(x, y, 0, x, y, r);
    grad.addColorStop(0, 'rgba(255,255,255,0.95)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(x - r, y - r, r * 2, r * 2);
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function makeGrassTexture() {
  const cv = document.createElement('canvas');
  cv.width = 512; cv.height = 512;
  const g = cv.getContext('2d');
  const rng = makeRng(20260926);
  g.fillStyle = '#6fa35f';
  g.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 8; i++) { // 割草条纹(沿跑道方向)
    if (i % 2) { g.fillStyle = 'rgba(255,255,255,0.05)'; g.fillRect(0, i * 64, 512, 64); }
  }
  for (let i = 0; i < 900; i++) { // 杂色草斑
    const x = rng() * 512, y = rng() * 512, r = 1 + rng() * 3;
    g.fillStyle = rng() < 0.5 ? 'rgba(60,110,50,0.16)' : 'rgba(150,190,110,0.13)';
    g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(24, 24);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}
