// 直线跑回归测试: 在 Node 里复现与页面 stepOnce 相同的闭环
// (MuJoCo WASM + ONNX 策略 + 航向保持外环), 验证每台机器人都能沿本车道跑完 25m。
// 运行: node test-straightline.mjs
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require_ = createRequire(import.meta.url);
const ROOT = path.dirname(fileURLToPath(import.meta.url));
const imp = (p) => import(pathToFileURL(path.join(ROOT, p)).href);

const ortSrc = fs.readFileSync(path.join(ROOT, 'vendor/ort/ort.min.js'), 'utf8');
const ort = new Function('require', ortSrc + '\n;return ort;')(require_);
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = pathToFileURL(path.join(ROOT, 'vendor/ort')).href + '/';

const loadMujoco = (await imp('vendor/mujoco/mujoco.js')).default;
const mj = await loadMujoco();
const { Sim, buildSceneXml, makeRng } = await imp('src/sim.js');
const { PolicyRunner, PolicySession, CFG, steerCmd } = await imp('src/policy.js');

const assets = {
  xml: fs.readFileSync(path.join(ROOT, 'assets/g1_29dof.xml'), 'utf8'),
  meshes: new Map(),
  sceneXml: buildSceneXml(),
};
for (const name of [...new Set([...assets.xml.matchAll(/file="([^"]+\.STL)"/g)].map((m) => m[1]))]) {
  assets.meshes.set(name, new Uint8Array(fs.readFileSync(path.join(ROOT, 'assets/meshes', name))));
}
const sim = await Sim.load(mj, assets, () => {});
const session = new PolicySession(ort);
await session.load(new Uint8Array(fs.readFileSync(path.join(ROOT, 'assets/policy.onnx'))));

// 单台机器人跑完整赛程: 前 1.2s 站立(对应倒计时), 之后指令同 main.js stepOnce
async function raceRobot(seed, laneY, speed) {
  const robot = { data: new mj.MjData(sim.model) };
  const runner = new PolicyRunner();
  sim.resetRobot(robot, laneY, makeRng(seed), 0.025, 0);
  const cmd = new Float32Array(3);
  for (let s = 0; s < Math.ceil(60 / 0.02); s++) {
    if (s === 0 || s === 60) cmd[0] = speed;
    cmd[2] = steerCmd(robot.data.qpos, laneY);
    runner.pushObs(runner.buildObs(robot.data.qpos, robot.data.qvel, cmd));
    const out = await session.inferAll([runner]);
    runner.applyAction(out[0]);
    for (let k = 0; k < CFG.decimation; k++) {
      runner.pd(robot.data.qpos, robot.data.qvel, robot.data.ctrl);
      mj.mj_step(sim.model, robot.data);
    }
    if (robot.data.qpos[0] >= 25) return { x: robot.data.qpos[0], yErr: robot.data.qpos[1] - laneY, t: s * 0.02, ok: true };
    if (robot.data.qpos[2] < 0.45) return { x: robot.data.qpos[0], yErr: robot.data.qpos[1] - laneY, t: s * 0.02, ok: false };
  }
  return { x: robot.data.qpos[0], yErr: robot.data.qpos[1] - laneY, t: 60, ok: false };
}

// 6 条道各跑一台, 速度覆盖 0.5~1.55 m/s(全程稳定包线, 上限见 test-speed-sweep.mjs)
const lanes = [-3.375, -2.025, -0.675, 0.675, 2.025, 3.375];
const speeds = [0.5, 0.8, 1.0, 1.2, 1.4, 1.55];
let pass = true;
console.log('直线跑回归测试(航向保持外环生效):');
for (let i = 0; i < 6; i++) {
  const r = await raceRobot(1000 + i * 77, lanes[i], speeds[i]);
  const ok = r.ok && Math.abs(r.yErr) < 0.35;
  pass = pass && ok;
  console.log(`  ${ok ? '✓' : '✗'} 道次${i + 1}(y=${lanes[i]}), v=${speeds[i]}: ${r.ok ? '完赛' : '未完赛'} t=${r.t.toFixed(1)}s 末端横向偏差=${r.yErr.toFixed(2)}m`);
}
console.log(pass ? 'ALL PASS: 全部沿本车道完赛(偏差 < 0.35m)' : 'FAILED');
process.exit(pass ? 0 : 1);
