// G1 短跑大赛 · 主程序
// MuJoCo WASM 物理 + ONNX RL 策略 + three.js 渲染, 全部在浏览器本地运行。

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { CFG, POLICY_DT, PolicyRunner, PolicySession } from './policy.js';
import { Sim, buildSceneXml, makeRng } from './sim.js';
import { buildTrack, addLights, RobotVisual, TEAM_COLORS, laneY, MAX_LANES } from './scene.js';
import { Race, RACE_TIMEOUT } from './race.js';

const $ = (id) => document.getElementById(id);
const bootlog = $('bootlog');
function log(msg, cls = '') {
  const div = document.createElement('div');
  div.textContent = msg;
  if (cls) div.className = cls;
  bootlog.appendChild(div);
  bootlog.scrollTop = bootlog.scrollHeight;
  $('bootbar').value = Math.min(99, $('bootbar').value + 6);
}

const app = {
  mujoco: null,
  sim: null,
  session: null,
  robots: [],      // { sim, runner, visual, cmd, laneY, name, color, targetSpeed, ...race字段 }
  race: null,
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  sun: null,
  camMode: 'leader',
  simSpeed: 1,
  paused: false,
  baseSpeed: 0.9,  // m/s(该策略训练范围约 0~1.0)
  robotCount: 4,
  seed: 20260923,
  rtf: 0,          // realtime factor
};

// ---------- 资源加载 ----------
async function fetchAssets() {
  log('获取 G1 MJCF ...');
  const xml = await (await fetch('./assets/g1_29dof.xml')).text();
  // XML 中 mesh 引用为 file="xxx.STL", 目录由 meshdir="meshes" 指定
  const meshFiles = [...new Set([...xml.matchAll(/file="([^"]+\.STL)"/g)].map((m) => m[1]))];
  log(`发现 ${meshFiles.length} 个网格文件, 加载(本地缓存)...`);
  const meshes = new Map();
  await Promise.all(meshFiles.map(async (name) => {
    const buf = await (await fetch('./assets/meshes/' + name)).arrayBuffer();
    meshes.set(name, new Uint8Array(buf));
  }));
  log('获取 ONNX 策略(1.7MB) ...');
  const onnx = await (await fetch('./assets/policy.onnx')).arrayBuffer();
  return { xml, meshes, onnx };
}

// ---------- 机器人构建 ----------
function buildRobots(n) {
  // 清理旧的
  for (const r of app.robots) {
    app.scene.remove(r.visual.group);
    r.visual.dispose();
    try { r.sim.data.delete(); } catch (e) { /* 忽略 */ }
  }
  app.robots = [];
  for (let i = 0; i < n; i++) {
    const simRobot = app.sim.addRobot();
    const runner = new PolicyRunner();
    const color = TEAM_COLORS[i % TEAM_COLORS.length];
    const visual = new RobotVisual(app.mujoco, app.sim.model, color, i + 1);
    app.scene.add(visual.group);
    app.robots.push({
      sim: simRobot,
      runner,
      visual,
      cmd: new Float32Array(3),
      laneY: laneY(i),
      name: `${i + 1}号`,
      color,
      targetSpeed: 0,
      finished: false, finishTime: 0, penalty: 0, falls: 0,
      fallen: false, fallenAt: 0, x: 0, speed: 0, place: 0,
    });
  }
  app.race.robots = app.robots;
  applySpeeds();
  app.race.resetAll(app.seed);
}

function applySpeeds() {
  const rng = makeRng(app.seed ^ 0x9e3779b9);
  for (const r of app.robots) {
    // 每台机器人的目标速度略有差别(像真实比赛的不同配速策略), 并限制在策略训练范围内
    r.targetSpeed = Math.max(0.2, Math.min(1.0, app.baseSpeed + (rng() * 2 - 1) * 0.1));
    if (app.race.state === 'racing') r.cmd[0] = r.targetSpeed;
  }
}

// ---------- 仿真步进 ----------
let acc = 0;
let busy = false;
let simAdvanced = 0;
let realAccum = 0;

async function stepOnce() {
  const robots = app.robots;
  // 构建观测
  for (const r of robots) {
    if (r.fallen) continue;
    r.runner.pushObs(r.runner.buildObs(r.sim.data.qpos, r.sim.data.qvel, r.cmd));
  }
  // 推理
  const actives = robots.filter((r) => !r.fallen);
  if (actives.length > 0) {
    const outs = await app.session.inferAll(actives.map((r) => r.runner));
    actives.forEach((r, i) => r.runner.applyAction(outs[i]));
  }
  // 500Hz 物理子步
  for (let k = 0; k < CFG.decimation; k++) {
    for (const r of robots) {
      if (r.fallen) continue;
      r.runner.pd(r.sim.data.qpos, r.sim.data.qvel, r.sim.data.ctrl);
      app.sim.step(r);
    }
  }
  app.race.tick(POLICY_DT);
}

// ---------- 相机 ----------
const camPos = new THREE.Vector3(-6, 0, 3);
const camAim = new THREE.Vector3(0, 0, 1);

function updateCamera(dt) {
  const leader = app.robots.reduce((best, r) => (!best || r.x > best.x ? r : best), null);
  const lx = leader ? leader.x : 0;
  const ly = leader ? leader.laneY : 0;
  if (app.camMode === 'free') {
    app.controls.enabled = true;
    app.controls.update();
    return;
  }
  app.controls.enabled = false;
  let want, aim;
  if (app.camMode === 'leader') {
    want = new THREE.Vector3(lx - 5.0, ly * 0.4, 2.4);
    aim = new THREE.Vector3(lx + 2.2, ly * 0.5, 0.9);
  } else { // 全景
    want = new THREE.Vector3(lx + 1.5, -12.5, 8.0);
    aim = new THREE.Vector3(lx + 2, 0, 0.8);
  }
  const k = 1 - Math.exp(-dt * 3.5);
  camPos.lerp(want, k);
  camAim.lerp(aim, k);
  app.camera.position.copy(camPos);
  app.camera.lookAt(camAim);
}

// ---------- HUD ----------
function fmtTime(t) {
  if (!t && t !== 0) return '-';
  const m = Math.floor(t / 60), s = t - m * 60;
  return m > 0 ? `${m}:${s.toFixed(2).padStart(5, '0')}` : `${s.toFixed(2)}s`;
}

let hudTick = 0;
function updateHUD() {
  const st = app.race.state;
  const pill = $('status');
  if (st === 'ready') { pill.textContent = '就绪'; pill.className = 'pill warn'; }
  else if (st === 'countdown') { pill.textContent = '即将开始'; pill.className = 'pill warn'; }
  else if (st === 'racing') { pill.textContent = '比赛进行中'; pill.className = 'pill ok'; }
  else { pill.textContent = '已完赛'; pill.className = 'pill bad'; }
  $('clock').textContent = app.race.state === 'countdown' ? '00.0s' : `${app.race.raceClock.toFixed(1)}s`;
  $('rt').textContent = `RTF ${app.rtf.toFixed(2)} · 推理 ${app.session ? app.session.inferMs.toFixed(2) : '0'}ms`;

  // 大字倒计时
  const cd = $('countdown');
  if (st === 'countdown' && app.race.countdown > 1.0) {
    cd.style.display = 'block';
    cd.textContent = String(Math.ceil(app.race.countdown - 1));
  } else if (st === 'countdown') {
    cd.style.display = 'block';
    cd.textContent = 'GO!';
  } else {
    cd.style.display = 'none';
  }

  // 排名榜(每 3 帧刷新一次)
  if (++hudTick % 3 !== 0) return;
  const rows = app.race.standings();
  const lb = $('lb');
  lb.innerHTML = '';
  for (const r of rows) {
    const div = document.createElement('div');
    div.className = 'row' + (r.finished ? ' done' : '');
    const prog = Math.max(0, Math.min(100, (r.x / 25) * 100));
    const stateTxt = r.finished
      ? `🏁 第${r.place}名 ${fmtTime(r.finishTime)}`
      : r.fallen ? '😵 摔倒-扶起中' : `${r.speed.toFixed(2)} m/s`;
    div.innerHTML = `
      <span class="dot" style="background:${r.color}"></span>
      <span class="nm">${r.name}</span>
      <span class="bar"><i style="width:${prog}%"></i></span>
      <span class="st">${stateTxt}</span>`;
    lb.appendChild(div);
  }
}

function showResults() {
  const rows = app.race.standings();
  const tb = $('res-rows');
  tb.innerHTML = '';
  const medal = ['🥇', '🥈', '🥉'];
  rows.forEach((r, i) => {
    const tr = document.createElement('tr');
    const place = r.finished ? (medal[r.place - 1] || `${r.place}`) : 'DNF';
    const time = r.finished ? fmtTime(r.finishTime) : `跑了 ${r.x.toFixed(1)}m`;
    const avg = r.finished ? (25 / r.finishTime).toFixed(2) : '-';
    tr.innerHTML = `<td>${place}</td><td><span class="dot" style="background:${r.color}"></span>${r.name}</td>
      <td>${time}</td><td>${avg}</td><td>${r.falls}</td>`;
    tb.appendChild(tr);
  });
  $('results').classList.add('show');
}

// ---------- 主循环 ----------
// 仿真与渲染解耦: 仿真由 4ms 定时器驱动(rAF 在部分环境会被冻结),
// 渲染走 rAF, rAF 不触发时用 100ms 定时器兜底, 保证任何环境下都能前进。
let simLastT = performance.now();

function simTick() {
  if (busy || app.paused || !app.session) { simLastT = performance.now(); return; }
  const now = performance.now();
  const dt = Math.min((now - simLastT) / 1000, 0.1);
  simLastT = now;
  realAccum += dt;
  if (realAccum > 0.5) { app.rtf = simAdvanced / realAccum; simAdvanced = 0; realAccum = 0; }
  busy = true;
  (async () => {
    try {
      acc += dt * app.simSpeed;
      let n = 0;
      while (acc >= POLICY_DT && n < 8) {
        await stepOnce();
        acc -= POLICY_DT; simAdvanced += POLICY_DT; n++;
      }
      if (acc > 0.5) acc = 0;
    } catch (e) {
      console.error('step error', e);
      log('仿真出错: ' + e.message, 'err');
      app.paused = true;
    } finally {
      busy = false;
    }
  })();
}

let lastRenderT = 0;
function renderTick() {
  if (!app.renderer || !app.race || !app.session) return;
  const now = performance.now();
  const dt = Math.min((now - lastRenderT) / 1000 || 0.016, 0.1);
  lastRenderT = now;
  for (const r of app.robots) r.visual.update(r.sim.data);
  updateCamera(dt);
  updateHUD();
  // 阳光跟随领跑者, 保证阴影覆盖
  const lx = app.robots.reduce((b, r) => (!b || r.x > b.x ? r : b), null)?.x ?? 0;
  app.sun.position.set(lx - 10, -8, 20);
  app.sun.target.position.set(lx, 0, 0);
  app.renderer.render(app.scene, app.camera);
  if (app.race.state === 'finished' && !$('results').classList.contains('show')) showResults();
}

function frame() {
  requestAnimationFrame(frame);
  lastRenderT = performance.now();
  renderTick();
}
// rAF 冻结时的渲染兜底(约 10fps)
let renderFallbackStarted = false;
function startLoops() {
  if (renderFallbackStarted) return;
  renderFallbackStarted = true;
  setInterval(simTick, 4);
  setInterval(() => {
    if (performance.now() - lastRenderT > 120) {
      lastRenderT = performance.now();
      renderTick();
    }
  }, 100);
  requestAnimationFrame(frame);
}

// ---------- UI 事件 ----------
function wireUI() {
  $('btn-start').onclick = () => {
    $('results').classList.remove('show');
    app.seed = (Math.random() * 0xffffffff) >>> 0;
    app.race.newSeed(app.seed);
    applySpeeds();
    app.race.start();
  };
  $('btn-again').onclick = () => {
    $('results').classList.remove('show');
    app.seed = (Math.random() * 0xffffffff) >>> 0;
    app.race.newSeed(app.seed);
    applySpeeds();
    app.race.start();
  };
  $('res-close').onclick = () => $('results').classList.remove('show');

  for (const b of document.querySelectorAll('#count-seg button')) {
    b.onclick = () => {
      document.querySelectorAll('#count-seg button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      app.robotCount = parseInt(b.dataset.n, 10);
      buildRobots(app.robotCount);
    };
  }
  $('spd').oninput = (e) => {
    app.baseSpeed = parseFloat(e.target.value);
    $('spd-val').textContent = app.baseSpeed.toFixed(2) + ' m/s';
    applySpeeds();
  };
  for (const b of document.querySelectorAll('#speed-seg button')) {
    b.onclick = () => {
      document.querySelectorAll('#speed-seg button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      app.simSpeed = parseFloat(b.dataset.v);
    };
  }
  for (const b of document.querySelectorAll('#cam-seg button')) {
    b.onclick = () => {
      document.querySelectorAll('#cam-seg button').forEach((x) => x.classList.remove('active'));
      b.classList.add('active');
      app.camMode = b.dataset.m;
      if (app.camMode === 'free') {
        const leader = app.robots.reduce((a, r) => (!a || r.x > a.x ? r : a), null);
        app.controls.target.set(leader ? leader.x : 0, 0, 0.8);
        app.controls.update();
      }
    };
  }
  $('btn-pause').onclick = () => {
    app.paused = !app.paused;
    $('btn-pause').textContent = app.paused ? '继续' : '暂停';
  };
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); $('btn-start').click(); }
  });
}

// ---------- 启动 ----------
async function main() {
  try {
    log('加载 MuJoCo WASM(约 10MB, 首次稍慢)...');
    app.mujoco = await loadMujoco();
    log('MuJoCo ' + '就绪');

    const assets = await fetchAssets();
    app.sim = await Sim.load(app.mujoco, { xml: assets.xml, meshes: assets.meshes, sceneXml: buildSceneXml() }, log);

    log('初始化 three.js 场景 ...');
    const canvasWrap = $('view');
    app.renderer = new THREE.WebGLRenderer({ antialias: true });
    app.renderer.setSize(window.innerWidth, window.innerHeight);
    app.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    app.renderer.shadowMap.enabled = true;
    app.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    app.renderer.outputColorSpace = THREE.SRGBColorSpace;
    canvasWrap.appendChild(app.renderer.domElement);

    app.scene = new THREE.Scene();
    app.scene.background = new THREE.Color(0x9ec7e8);
    app.scene.fog = new THREE.Fog(0x9ec7e8, 40, 120);
    app.camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 400);
    app.camera.up.set(0, 0, 1); // MuJoCo z-up
    app.camera.position.copy(camPos);
    app.controls = new OrbitControls(app.camera, app.renderer.domElement);
    app.controls.enableDamping = true;
    app.controls.target.set(0, 0, 0.8);
    app.sun = addLights(app.scene);
    buildTrack(app.scene);

    log('加载 ONNX 推理会话 ...');
    const ort = window.ort;
    ort.env.wasm.wasmPaths = '/vendor/ort/';
    ort.env.wasm.numThreads = 1;
    app.session = new PolicySession(ort);
    const session = await app.session.load(assets.onnx);
    log(`策略就绪: 输入 ${session.inputNames[0]}(${app.session.batched ? '支持批量' : '逐台'}), 输出 ${session.outputNames[0]}`);

    app.race = new Race([], app.sim, null);
    buildRobots(app.robotCount);
    wireUI();

    window.addEventListener('resize', () => {
      app.camera.aspect = window.innerWidth / window.innerHeight;
      app.camera.updateProjectionMatrix();
      app.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    $('bootbar').value = 100;
    log('一切就绪! 点击「开始比赛」发枪 🏁');
    setTimeout(() => $('boot').classList.add('hidden'), 600);
    startLoops();
    // 自动化测试钩子
    window.__app = app;
    // 确定性推进: 不依赖定时器, 手动步进 N 个策略周期(页面被挂起时测试用)
    window.__kick = async (steps) => {
      for (let i = 0; i < steps; i++) {
        await stepOnce();
        simAdvanced += POLICY_DT;
      }
      renderTick();
      return window.__debug();
    };
    window.__debug = () => ({
      state: app.race.state, clock: app.race.raceClock, rtf: app.rtf,
      robots: app.robots.map((r) => {
        const q = r.sim.data.qpos;
        return {
          x: +(q[0] ?? NaN).toFixed?.(2) ?? NaN,
          z: +(q[2] ?? NaN).toFixed?.(2) ?? NaN,
          v: +(r.sim.data.qvel[0] ?? NaN).toFixed?.(2) ?? NaN,
          qposLen: q.length,
          falls: r.falls, done: r.finished,
        };
      }),
    });
  } catch (e) {
    console.error(e);
    log('启动失败: ' + (e.message || e), 'err');
    log('请确认通过本地 HTTP 服务访问本页(如 python -m http.server)。', 'err');
  }
}

main();
