// 策略观测构建、ONNX 会话封装与 PD 控制 —— 按物种契约参数化。
//
// 三种观测布局(contract.layout), 均逐条对齐官方部署代码:
//  · 'g1'  (unitree_rl_gym): obs(96) = [机体角速度×0.2(3), 重力投影(3), 指令(3),
//          关节位置-默认(29), 关节速度×0.05(29), 上次动作(29)],
//          组优先堆叠 5 帧 = 480; 策略顺序≠mujoco 顺序(policyToXml)。
//  · 'sa01'(engineai sim2sim): obs(47) = [sin(2πt/T), cos(2πt/T), vx·2, vy·2, wy·1,
//          关节位置-默认(12), 关节速度×0.05(12), 上次动作(12), 机体角速度(3), 欧拉角(3)],
//          帧优先堆叠 15 帧 = 705, 策略周期 10ms。
//  · 't1'  (booster deploy): obs(47) = [重力投影(3), 机体角速度(3), 指令(3),
//          cos(2πφ), sin(2πφ), 关节位置-默认(12), 关节速度×0.1(12), 上次动作(12)],
//          单帧, 指令平滑+步态门控, 策略周期 20ms。
// 输出动作 -> 目标角 = a·actionScale + 默认角; 关节空间 PD(带力矩限幅)。
// 比赛直道保持: 上层横向 PD 级联外环把车道偏差转成 cmd[2], 见 STEER。

// ---------- G1 契约(布局细节与 G1 部署管线绑定) ----------
export const CFG = {
  layout: 'g1',
  numActions: 29,
  numObs: 96,
  stack: 5,
  stackMode: 'group',
  angVelScale: 0.2,
  dofVelScale: 0.05,
  actionScale: 0.25,
};

export const NA = CFG.numActions;
export const POLICY_DT = 0.02; // G1 策略周期(其他物种见各自 dt×decim)

// G1 默认站姿(策略顺序)与顺序映射; DEFAULT_MJC 为 mujoco(执行器)顺序
const DEFAULT_ANGLES_POLICY = [-0.1, -0.1, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, -0.2, -0.2, 0.25, -0.25, 0, 0, 0, 0, 0.97, 0.97, 0.15, -0.15, 0, 0, 0, 0];
export const POLICY_TO_XML = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28];
export const DEFAULT_MJC = POLICY_TO_XML.map((pi) => DEFAULT_ANGLES_POLICY[pi]);
const XML_TO_POLICY = [];
for (let i = 0; i < NA; i++) XML_TO_POLICY[POLICY_TO_XML[i]] = i;

// 重力投影: 世界重力在机体坐标系下的表示(由四元数直接构造, 直立时≈[0,0,-1])
function gravityOrientation(qw, qx, qy, qz) {
  return [
    2 * (-qz * qx + qw * qy),
    -2 * (qz * qy + qw * qx),
    1 - 2 * (qw * qw + qz * qz),
  ];
}

// 四元数(w,x,y,z) -> 欧拉角 roll/pitch/yaw(与 engineai sim2sim 的 quaternion_to_euler_array 一致)
function quatToRpy(qw, qx, qy, qz) {
  const roll = Math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy));
  let pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (qw * qy - qz * qx))));
  const yaw = Math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
  pitch = ((pitch + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return [roll, pitch, yaw];
}

// ---------- 航向保持外环(横向 PD 级联) ----------
// 速度策略只跟踪机体系速度指令, 对世界系航向没有反馈。外环分两级:
//  1) 横向: 车道偏差 P + 横向速度阻尼 D -> 期望横向速度(限幅), 与前进速度
//     合成期望航向角(限幅, 低速时不疯狂转向);
//  2) 航向: 期望航向与实际航向的误差 P + 偏航角速度阻尼 D -> cmd[2]。
// 纯"瞄准前方一点"的 P 控制在高速下欠阻尼会画 S 形, 这里两处阻尼把它压住。
export const STEER = {
  kpLat: 1.1,      // 横向偏差 -> 期望横向速度 (1/s)
  kdLat: 0.9,      // 横向速度阻尼 (1/s)
  maxLat: 0.30,    // 期望横向速度限幅 (m/s), 避免超出策略训练分布
  maxHeadErr: 0.7, // 期望航向角限幅 (rad)
  kpYaw: 2.2,      // 航向误差 -> yaw 角速度指令 (1/s)
  kdYaw: 0.12,     // 偏航角速度阻尼
  maxYawCmd: 1.0,  // yaw 指令绝对限幅(再受物种 yawCap 约束)
};

export function headingYaw(qw, qx, qy, qz) {
  return Math.atan2(2 * (qx * qy + qw * qz), 1 - 2 * (qy * qy + qz * qz));
}

// 物种级覆盖: makeSteer({kpYaw: 1.5}) -> 与 STEER 默认合并后的完整参数对象
export function makeSteer(over = {}) {
  return { ...STEER, ...over };
}

export function wrapPi(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

export function steerCmd(q, qvel, laneY, steer = STEER) {
  const clampV = (x, a, b) => Math.max(a, Math.min(b, x));
  const e = q[1] - laneY;                            // 横向偏差(>0: 车道左侧)
  const vy = qvel ? qvel[1] : 0;                     // 世界系横向速度
  const vx = Math.max(0.15, qvel ? qvel[0] : 1.0);   // 前进速度(下限防低速发散)
  const vLat = clampV(-steer.kpLat * e - steer.kdLat * vy, -steer.maxLat, steer.maxLat);
  const desired = clampV(Math.atan2(vLat, vx), -steer.maxHeadErr, steer.maxHeadErr);
  const err = wrapPi(desired - headingYaw(q[3], q[4], q[5], q[6]));
  const wz = qvel ? qvel[5] : 0;                     // 机体系偏航角速度
  return clampV(steer.kpYaw * err - steer.kdYaw * wz, -steer.maxYawCmd, steer.maxYawCmd);
}

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// ---------- 策略运行状态 ----------
export class PolicyRunner {
  constructor(contract = CFG) {
    this.c = contract;
    this.NA = contract.numActions;
    this.period = contract.period ?? 0.02; // 策略周期(秒)
    // 目标角缓冲(mujoco 顺序)与默认角
    if (contract.layout === 'g1') {
      this.defaultMjc = Float32Array.from(DEFAULT_MJC);
      this.policyToXml = POLICY_TO_XML;
    } else {
      this.defaultMjc = Float32Array.from(contract.defaultDof);
      this.policyToXml = Array.from({ length: this.NA }, (_, i) => i);
    }
    this.action = new Float32Array(this.NA);
    this.target = Float32Array.from(this.defaultMjc);
    this.input = new Float32Array(contract.inputLen ?? contract.numObs);
    // sa01 帧优先堆叠 / g1 组优先堆叠
    if (contract.stackMode !== 'none') this.frames = [];
    // t1 指令平滑状态
    if (contract.layout === 't1') {
      this.smoothed = new Float32Array(3);
      this.gaitActive = 0;
      this.phase = 0;
    }
    this.obsTime = 0; // 策略时钟(步态相位用)
    this.reset();
  }

  reset() {
    this.action.fill(0);
    this.target.set(this.defaultMjc);
    this.input.fill(0);
    this.obsTime = 0;
    if (this.c.stackMode !== 'none') {
      this.frames = [];
      for (let i = 0; i < this.c.stack; i++) this.frames.push(new Float32Array(this.c.stackMode === 'pm01' ? 75 : (this.c.stackMode === 'frame' ? this.c.numSingleObs : this.c.numObs)));
    }
    if (this.c.layout === 't1') {
      this.smoothed.fill(0);
      this.gaitActive = 0;
      this.phase = 0;
    }
  }

  // 构建单帧观测并打包进策略输入。cmd = [vx, vy, wy](物理单位), q 为 mujoco qpos。
  buildAndPushObs(qpos, qvel, cmd, dtStep) {
    const c = this.c;
    let single;
    if (c.layout === 'g1') {
      single = this._obsG1(qpos, qvel, cmd);
    } else if (c.layout === 'pm01') {
      this.obsTime += dtStep;
      single = this._obsPm01(qpos, qvel, cmd);
    } else if (c.layout === 'sa01') {
      this.obsTime += dtStep;
      single = this._obsSa01(qpos, qvel, cmd);
    } else {
      this.obsTime += dtStep;
      single = this._obsT1(qpos, qvel, cmd, dtStep);
    }
    if (c.stackMode === 'pm01') {
      // PM01: 各观测项(关节位置/速度/上次动作/角速度/重力)各自 15 帧历史, 项优先拼接, 末尾接当前指令
      this.frames.push(single);
      if (this.frames.length > c.stack) this.frames.shift();
      const dims = [23, 23, 23, 3, 3];
      let base = 0;
      let tOff = 0;
      for (const d of dims) {
        for (const f of this.frames) {
          for (let i = 0; i < d; i++) this.input[base + i] = f[tOff + i];
          base += d;
        }
        tOff += d;
      }
      this.input[base] = cmd[0]; this.input[base + 1] = cmd[1]; this.input[base + 2] = cmd[2];
    } else if (c.stackMode === 'frame') {
      this.frames.push(single);
      if (this.frames.length > c.stack) this.frames.shift();
      let o = 0;
      for (const f of this.frames) { this.input.set(f, o); o += c.numSingleObs; }
    } else if (c.stackMode === 'group') {
      // G1: 组优先打包(每个观测分量组在帧维上连续)
      this.frames.push(single);
      if (this.frames.length > c.stack) this.frames.shift();
      const NA2 = c.numActions;
      const bounds = [[0, 3], [3, 6], [6, 9], [9, 9 + NA2], [9 + NA2, 9 + 2 * NA2], [9 + 2 * NA2, 9 + 3 * NA2]];
      let o = 0;
      for (const [a, b] of bounds) {
        for (const f of this.frames) {
          for (let i = a; i < b; i++) this.input[o++] = f[i];
        }
      }
    } else if (c.stackMode === 'none') {
      this.input.set(single);
    }
    return this.input;
  }

  _obsG1(qpos, qvel, cmd) {
    const obs = new Float32Array(this.c.numObs);
    const g = gravityOrientation(qpos[3], qpos[4], qpos[5], qpos[6]);
    for (let i = 0; i < 3; i++) {
      obs[i] = qvel[3 + i] * this.c.angVelScale;
      obs[3 + i] = g[i];
      obs[6 + i] = cmd[i];
    }
    for (let i = 0; i < NA; i++) {
      const x = XML_TO_POLICY[i];
      obs[9 + i] = qpos[7 + x] - DEFAULT_MJC[x];
      obs[9 + NA + i] = qvel[6 + x] * this.c.dofVelScale;
      obs[9 + 2 * NA + i] = this.action[i];
    }
    return obs;
  }

  _obsPm01(qpos, qvel, cmd) {
    const c = this.c;
    const s = new Float32Array(75); // 23 pos + 23 vel + 23 act + 3 ang + 3 grav(全部原始值)
    for (let i = 0; i < c.numActions; i++) {
      s[i] = qpos[7 + i] - this.defaultMjc[i];
      s[23 + i] = qvel[6 + i];
      s[46 + i] = this.action[i];
    }
    for (let i = 0; i < 3; i++) s[69 + i] = qvel[3 + i];
    const g = gravityOrientation(qpos[3], qpos[4], qpos[5], qpos[6]);
    for (let i = 0; i < 3; i++) s[72 + i] = g[i];
    return s;
  }

  _obsSa01(qpos, qvel, cmd) {
    const c = this.c;
    const s = new Float32Array(c.numSingleObs);
    const ph = 2 * Math.PI * (this.obsTime / c.cycleTime);
    s[0] = Math.sin(ph);
    s[1] = Math.cos(ph);
    s[2] = cmd[0] * c.cmdScales.vx;
    s[3] = cmd[1] * c.cmdScales.vy;
    s[4] = cmd[2] * c.cmdScales.wy;
    for (let i = 0; i < c.numActions; i++) {
      s[5 + i] = qpos[7 + i] - this.defaultMjc[i];
      s[17 + i] = qvel[6 + i] * c.dofVelScale;
      s[29 + i] = this.action[i];
    }
    for (let i = 0; i < 3; i++) s[41 + i] = qvel[3 + i]; // 机体角速度(机体系)
    const rpy = quatToRpy(qpos[3], qpos[4], qpos[5], qpos[6]);
    for (let i = 0; i < 3; i++) s[44 + i] = rpy[i];
    for (let i = 0; i < c.numSingleObs; i++) s[i] = clamp(s[i], -c.clipObs, c.clipObs);
    return s;
  }

  _obsT1(qpos, qvel, cmd, dtStep) {
    const c = this.c;
    // 指令平滑(每策略周期变化量限幅 ±period) + 步态门控
    const interval = this.period;
    for (let i = 0; i < 3; i++) {
      const d = clamp(cmd[i] - this.smoothed[i], -interval, interval);
      this.smoothed[i] += d;
    }
    const norm = Math.hypot(this.smoothed[0], this.smoothed[1], this.smoothed[2]);
    this.gaitActive = norm < 1e-5 ? 0 : 1;
    this.phase = (this.obsTime * c.gaitFrequency) % 1;

    const s = new Float32Array(c.numObs);
    const g = gravityOrientation(qpos[3], qpos[4], qpos[5], qpos[6]);
    for (let i = 0; i < 3; i++) s[i] = g[i] * (c.gravityScale ?? 1);
    for (let i = 0; i < 3; i++) s[3 + i] = qvel[3 + i] * c.angVelScale;
    s[6] = this.smoothed[0] * c.cmdScales.vx * this.gaitActive;
    s[7] = this.smoothed[1] * c.cmdScales.vy * this.gaitActive;
    s[8] = this.smoothed[2] * c.cmdScales.wy * this.gaitActive;
    s[9] = Math.cos(2 * Math.PI * this.phase) * this.gaitActive;
    s[10] = Math.sin(2 * Math.PI * this.phase) * this.gaitActive;
    for (let i = 0; i < c.numActions; i++) {
      s[11 + i] = (qpos[7 + i] - this.defaultMjc[i]) * 1.0;
      s[23 + i] = qvel[6 + i] * c.dofVelScale;
      s[35 + i] = this.action[i];
    }
    return s;
  }

  // 推理返回的动作 -> 位置目标(mujoco 顺序)
  applyAction(rawAction) {
    const c = this.c;
    for (let i = 0; i < this.NA; i++) this.action[i] = c.clipAction === undefined ? rawAction[i] : clamp(rawAction[i], -c.clipAction, c.clipAction);
    for (let i = 0; i < this.NA; i++) {
      const sc = c.actionScales ? c.actionScales[this.policyToXml[i]] : c.actionScale;
      this.target[i] = this.action[this.policyToXml[i]] * sc + this.defaultMjc[i];
    }
  }

  // 关节空间 PD: 写 ctrl(执行器顺序), 带力矩限幅
  pd(qpos, qvel, ctrl) {
    const c = this.c;
    for (let i = 0; i < this.NA; i++) {
      const tau = c.kps[i] * (this.target[i] - qpos[7 + i]) - c.kds[i] * qvel[6 + i];
      const lim = Array.isArray(c.tauLimit) ? c.tauLimit[i] : c.tauLimit;
      ctrl[i] = clamp(tau, -lim, lim);
    }
  }
}

// ---------- ONNX 会话封装(按契约) ----------
// 若模型支持动态 batch 则一次前向跑完同契约的所有机器人, 否则逐台推理
export class PolicySession {
  constructor(ort, contract = CFG) {
    this.ort = ort;
    this.c = contract;
    this.session = null;
    this.batched = false;
    this.inferMs = 0;
  }

  async load(onnxBytes) {
    this.session = await this.ort.InferenceSession.create(onnxBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    // 实测一次批量前向, 判定模型是否接受动态 batch(元数据不一定可靠)
    this.batched = false;
    try {
      const probe = new this.ort.Tensor('float32', new Float32Array(2 * this.c.inputLen), [2, this.c.inputLen]);
      await this.session.run({ [this.session.inputNames[0]]: probe });
      this.batched = true;
    } catch (e) { /* 固定 batch=1, 走逐台推理 */ }
    return this.session;
  }

  get inputName() { return this.session.inputNames[0]; }

  // 单次前向透传(测试管线用)
  run(feed) { return this.session.run(feed); }

  // runners: PolicyRunner[]; 返回每台的 Float32Array(numActions) 动作
  async inferAll(runners) {
    const c = this.c;
    const NA2 = c.numActions;
    const t0 = performance.now();
    let outs;
    if (this.batched && runners.length > 1) {
      const n = runners.length;
      const batch = new Float32Array(n * c.numObs);
      for (let r = 0; r < n; r++) batch.set(runners[r].input, r * c.numObs);
      const res = await this.session.run({ [this.inputName]: new this.ort.Tensor('float32', batch, [n, c.inputLen]) });
      outs = this._extractBatch(res, n, NA2);
    } else {
      outs = new Array(runners.length);
      for (let r = 0; r < runners.length; r++) {
        const res = await this.session.run({ [this.inputName]: new this.ort.Tensor('float32', runners[r].input, [1, c.inputLen]) });
        outs[r] = this._firstFloat32(res);
      }
    }
    this.inferMs = performance.now() - t0;
    return outs;
  }

  _firstFloat32(res) {
    for (const k of Object.keys(res)) {
      const t = res[k];
      if (t && t.data instanceof Float32Array) return t.data;
    }
    return res[this.session.outputNames[0]].data;
  }

  _extractBatch(res, n, NA2) {
    let data = null;
    for (const k of Object.keys(res)) {
      const t = res[k];
      if (t && t.data instanceof Float32Array && t.data.length >= n * NA2) { data = t.data; break; }
    }
    const outs = new Array(n);
    for (let r = 0; r < n; r++) outs[r] = data.subarray(r * NA2, (r + 1) * NA2);
    return outs;
  }
}
