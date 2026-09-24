// G1 29dof 速度策略(onnx)的观测构建与动作后处理。
// 约定与 loco-lab / unitree_rl_gym 部署管线一致:
//   obs(96) = [基础角速度*0.2(3), 重力投影(3), 指令(3), 关节位置-默认(29), 关节速度*0.05(29), 上次动作(29)]
//   输入 = 组优先堆叠 5 帧: [角速度x5, 重力x5, 指令x5, 位置x5, 速度x5, 动作x5] = 480 维
//   输出 = 29 维缩放动作, 目标关节角 = 动作*0.25 + 默认角(策略顺序), 500Hz 关节空间 PD。
// 比赛直道保持: 上层纯跟踪外环把航向/横向误差转成 cmd[2](yaw 角速度), 见 STEER。

export const CFG = {
  timestep: 0.002,
  decimation: 10, // 50Hz 策略
  numActions: 29,
  numObs: 96,
  stack: 5,
  angVelScale: 0.2,
  dofVelScale: 0.05,
  actionScale: 0.25,
  kps: [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 200, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
  kds: [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
  // 策略关节顺序下的默认站姿
  defaultAnglesPolicyOrder: [-0.1, -0.1, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3, 0.3, -0.2, -0.2, 0.25, -0.25, 0, 0, 0, 0, 0.97, 0.97, 0.15, -0.15, 0, 0, 0, 0],
  // 策略顺序 <-> mujoco(xml/执行器)顺序 互查表
  policyToXml: [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
};

export const NA = CFG.numActions;
export const POLICY_DT = CFG.timestep * CFG.decimation; // 0.02s

// mujoco(xml)顺序下的默认角度
export const DEFAULT_MJC = CFG.policyToXml.map((pi) => CFG.defaultAnglesPolicyOrder[pi]);
// mujoco(xml)顺序 -> 策略顺序
export const XML_TO_POLICY = [];
for (let i = 0; i < NA; i++) XML_TO_POLICY[CFG.policyToXml[i]] = i;

// 重力投影: 世界重力在机体坐标系下的表示(由四元数直接构造)
function gravityOrientation(qw, qx, qy, qz) {
  return [
    2 * (-qz * qx + qw * qy),
    -2 * (qz * qy + qw * qx),
    1 - 2 * (qw * qw + qz * qz),
  ];
}

// ---------- 航向保持外环(纯跟踪式) ----------
// 速度策略只跟踪机体系速度指令, 对世界系航向没有任何反馈: 实测 cmd=(vx,0,0) 直行
// 25m 会系统性偏出 10m+(策略/模型的固有配平偏置)。外环瞄准本车道前方一点,
// 把航向误差转成策略的偏航角速度指令 cmd[2], 补上这个反馈。
export const STEER = {
  kp: 2.0,        // 航向误差 -> yaw 角速度指令(rad/s)
  lookahead: 2.0, // 前视距离(m): 兼顾收敛横向偏差
  maxYawCmd: 1.0, // 策略训练范围内的 yaw 指令上限
};

// 机体系 +x(前向)轴的世界系偏航角, 由机体四元数直接构造
export function headingYaw(qw, qx, qy, qz) {
  return Math.atan2(2 * (qx * qy + qw * qz), 1 - 2 * (qy * qy + qz * qz));
}

export function wrapPi(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

// 由位姿计算航向保持的 yaw 角速度指令(q 为 mujoco qpos)
export function steerCmd(q, laneY, steer = STEER) {
  const desired = Math.atan2(laneY - q[1], steer.lookahead);
  const err = wrapPi(desired - headingYaw(q[3], q[4], q[5], q[6]));
  return Math.max(-steer.maxYawCmd, Math.min(steer.maxYawCmd, steer.kp * err));
}

// 单台机器人的策略运行状态(帧堆叠 + 上次动作 + 位置目标)
export class PolicyRunner {
  constructor() {
    this.frames = [];
    this.action = new Float32Array(NA); // 策略顺序
    this.target = Float32Array.from(DEFAULT_MJC); // mujoco 顺序
    this.input = new Float32Array(CFG.stack * CFG.numObs); // 480
    this.reset();
  }

  reset() {
    this.frames = [];
    for (let i = 0; i < CFG.stack; i++) this.frames.push(new Float32Array(CFG.numObs));
    this.action.fill(0);
    this.target.set(DEFAULT_MJC);
    this.input.fill(0);
  }

  // 由 mujoco 状态构建 96 维观测(与 loco-lab 验证过的管线逐字一致)
  buildObs(qpos, qvel, cmd) {
    const obs = new Float32Array(CFG.numObs);
    const g = gravityOrientation(qpos[3], qpos[4], qpos[5], qpos[6]);
    for (let i = 0; i < 3; i++) {
      obs[i] = qvel[3 + i] * CFG.angVelScale; // 自由关节角速度(机体系)
      obs[3 + i] = g[i];
      obs[6 + i] = cmd[i];
    }
    for (let i = 0; i < NA; i++) {
      const x = XML_TO_POLICY[i];
      obs[9 + i] = qpos[7 + x] - DEFAULT_MJC[x]; // 关节位置
      obs[9 + NA + i] = qvel[6 + x] * CFG.dofVelScale; // 关节速度
      obs[9 + 2 * NA + i] = this.action[i]; // 上次动作
    }
    return obs;
  }

  pushObs(obs) {
    this.frames.push(obs);
    if (this.frames.length > CFG.stack) this.frames.shift();
    // 组优先打包: 每个观测分量组在帧维上连续
    let o = 0;
    const big = this.input;
    const bounds = [[0, 3], [3, 6], [6, 9], [9, 9 + NA], [9 + NA, 9 + 2 * NA], [9 + 2 * NA, 9 + 3 * NA]];
    for (const [a, b] of bounds) {
      for (const f of this.frames) {
        for (let i = a; i < b; i++) big[o++] = f[i];
      }
    }
  }

  // 推理返回的动作 -> 位置目标(mujoco 顺序)
  applyAction(rawAction) {
    for (let i = 0; i < NA; i++) this.action[i] = rawAction[i];
    for (let i = 0; i < NA; i++) {
      this.target[i] = this.action[CFG.policyToXml[i]] * CFG.actionScale + DEFAULT_MJC[i];
    }
  }

  // 关节空间 PD: 写 ctrl(mujoco 顺序)
  pd(qpos, qvel, ctrl) {
    for (let i = 0; i < NA; i++) {
      ctrl[i] = CFG.kps[i] * (this.target[i] - qpos[7 + i]) - CFG.kds[i] * qvel[6 + i];
    }
  }
}

// ONNX 会话封装: 若模型支持动态 batch 则一次前向跑完所有机器人, 否则逐台推理
export class PolicySession {
  constructor(ort) {
    this.ort = ort;
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
      const probe = new this.ort.Tensor('float32', new Float32Array(2 * CFG.numObs * CFG.stack), [2, CFG.numObs * CFG.stack]);
      await this.session.run({ [this.session.inputNames[0]]: probe });
      this.batched = true;
    } catch (e) { /* 固定 batch=1, 走逐台推理 */ }
    return this.session;
  }

  get inputName() { return this.session.inputNames[0]; }

  // runners: PolicyRunner[]; 返回每台的 Float32Array(29) 动作
  async inferAll(runners) {
    const t0 = performance.now();
    let outs;
    if (this.batched && runners.length > 1) {
      const n = runners.length;
      const batch = new Float32Array(n * CFG.numObs * CFG.stack);
      for (let r = 0; r < n; r++) batch.set(runners[r].input, r * CFG.numObs * CFG.stack);
      const res = await this.session.run({ [this.inputName]: new this.ort.Tensor('float32', batch, [n, CFG.numObs * CFG.stack]) });
      outs = this._extractBatch(res, n);
    } else {
      outs = new Array(runners.length);
      for (let r = 0; r < runners.length; r++) {
        const res = await this.session.run({ [this.inputName]: new this.ort.Tensor('float32', runners[r].input, [1, CFG.numObs * CFG.stack]) });
        outs[r] = this._firstFloat32(res);
      }
    }
    this.inferMs = performance.now() - t0;
    return outs;
  }

  _firstFloat32(res) {
    const key = this.session.outputNames[0];
    for (const k of Object.keys(res)) {
      const t = res[k];
      if (t && t.data instanceof Float32Array) return t.data;
    }
    return res[key].data;
  }

  _extractBatch(res, n) {
    let data = null;
    for (const k of Object.keys(res)) {
      const t = res[k];
      if (t && t.data instanceof Float32Array && t.data.length >= n * NA) { data = t.data; break; }
    }
    const outs = new Array(n);
    for (let r = 0; r < n; r++) outs[r] = data.subarray(r * NA, (r + 1) * NA);
    return outs;
  }
}
