// 机器人物种注册表 —— 只收录"官方模型文件 + 官方策略模型"齐备的真实机器人。
//
// 每个物种:
//   xmlFile/meshesDir  磁盘上的 MJCF 模型资产(官方仓库原版, 仅运行时打最小补丁)
//   policyFile         官方发布的策略权重(ONNX; TorchScript 权重已事先转换)
//   contract           该策略的观测/动作/PD 契约(逐条对齐官方部署脚本/配置)
//   dt×decim           仿真步长与策略频率(与官方部署一致)
// 模型与策略来源(调研定案 2026-09):
//   G1   unitree 官方 MJCF(unitree_ros) + 策略 RoboCubPilot/g1_deploy_mujoco
//   SA01 众擎 engineai_legged_gym(官方 MJCF+STL+ONNX+sim2sim 脚本)
//   T1   Booster booster_gym(官方 MJCF+T1.pt 权重+T1.yaml 部署配置)
// 落选记录: 智元 A2(官方仅 X2 URDF 无策略)、优必选 Walker S2(无公开模型+策略对)、
//   MicroDuck(无公开资产)、傅里叶 GR-1/N1(有模型有权重但 obs 契约锁在闭源 SDK/
//   多层配置里无法对齐, 硬接必摔)。

// ---------- G1 契约(unitree_rl_gym 布局, 见 policy.js) ----------
export const G1_CONTRACT = {
  layout: 'g1',
  numActions: 29,
  numObs: 96,
  inputLen: 480,
  stack: 5,
  stackMode: 'group',        // 组优先堆叠
  angVelScale: 0.2,
  dofVelScale: 0.05,
  actionScale: 0.25,
  kps: [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 200, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
  kds: [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
  tauLimit: 1e9, // 无额外限幅(力矩上限由 MJCF 的 actuatorfrcrange 把关)
};

// ---------- 众擎 PM01 契约(engineai_rl_lab amp_velocity_flat_pm01 逐条对齐) ----------
// 官方资产: GMR 仓库 MJCF(pm_v2) + rl_lab AMP 速度策略导出 ONNX + params/env.yaml。
// obs(1128) = [joint_pos(23), joint_vel(23), 上次动作(23), 角速度(3), 重力投影(3)]
//              各 15 帧历史(原始值, 无缩放) + 当前速度指令(3)。
// 动作 23 关节(J00..J22), 目标角 = 每关节缩放·a + 0(默认站姿为直立零位)。
// PD@500Hz, 策略@100Hz(decim 5 × dt 0.002); 指令范围 vx 0.5~0.8, yaw ±0.6。
export const PM01_CONTRACT = {
  layout: 'pm01',
  numActions: 23,
  numObs: 1128,
  inputLen: 1128,
  stack: 15,
  stackMode: 'pm01',
  actionScales: [0.5, 0.2, 0.2, 0.5, 0.5, 0.2, 0.5, 0.2, 0.2, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
  kps: [110, 70, 70, 110, 30, 30, 110, 70, 70, 110, 30, 30, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
  kds: [5, 3, 3, 5, 0.3, 0.3, 5, 3, 3, 5, 0.3, 0.3, 3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
  tauLimit: [164, 164, 61, 164, 54.9, 54.9, 164, 164, 61, 164, 54.9, 54.9, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61],
  defaultDof: new Array(23).fill(0),
};

// ---------- Booster T1 契约(booster_gym deploy/configs/T1.yaml + utils/policy.py 逐条对齐) ----------
// obs(47) = [重力投影(3), 机体角速度(3), vx, vy, wy, cos(2πφ), sin(2πφ),
//            (q-def)(12), dq·0.1(12), 上次动作(12)], 单帧无堆叠。
// 指令平滑(每策略周期 ±period 限幅) + 步态门控(|cmd|≈0 时步态项清零)。
// PD@500Hz, 策略@50Hz(decim 10 × dt 0.002), 动作限幅 ±1, 目标角 = a·1.0 + 默认角。
export const T1_CONTRACT = {
  layout: 't1',
  numActions: 12,
  numObs: 47,
  inputLen: 47,
  stack: 1,
  stackMode: 'none',
  gaitFrequency: 1.0,
  cmdScales: { vx: 1.0, vy: 1.0, wy: 1.0 },
  gravityScale: 1.0,
  angVelScale: 1.0,
  dofVelScale: 0.1,
  actionScale: 1.0,
  clipObs: 100.0,
  clipAction: 1.0,
  kps: [200, 200, 200, 200, 50, 50, 200, 200, 200, 200, 50, 50],
  kds: [5, 5, 5, 5, 3, 3, 5, 5, 5, 5, 3, 3],
  tauLimit: [60, 25, 30, 60, 24, 15, 60, 25, 30, 60, 24, 15],
  defaultDof: [-0.2, 0, 0, 0.4, -0.25, 0, -0.2, 0, 0, 0.4, -0.25, 0],
};

// ---------- 物种注册表 ----------
export const SPECIES = [
  {
    id: 'g1',
    name: 'Unitree G1',
    short: 'G1',
    emoji: '🤖',
    xmlFile: './assets/g1_29dof.xml',
    policyFile: './assets/policy.onnx',
    contract: G1_CONTRACT,
    dt: 0.002,
    decim: 10,
    zHome: 0.793,
    fallZ: 0.45,
    labelH: 1.25,
    visGroups: [1],
    maxV: 1.55,
    noise: 0.025,
    // G1 资产为"机器人 MJCF", 需包一层比赛场景
    wrapScene: true,
  },
  {
    id: 'pm01',
    name: '众擎 PM01',
    short: 'PM01',
    emoji: '🦾',
    xmlFile: './assets/pm01/pm_v2.xml',
    extraFiles: ['xml/serial_pm_v2.xml', 'xml/serial_links.xml', 'xml/serial_actuators.xml', 'xml/serial_sensors.xml', 'xml/assets.xml', 'ground.xml'],
    policyFile: './assets/pm01/policy.onnx',
    contract: PM01_CONTRACT,
    dt: 0.002,
    decim: 5,
    zHome: 0.75,
    fallZ: 0.35,
    labelH: 1.35,
    visGroups: [2],
    maxV: 0.8,
    noise: 0.01,
    yawCap: 0.6,
    wrapScene: false,
    // 官方地面 friction 0.6, 训练时物理材质为 1.0 -> 对齐
    xmlPatches: [{ from: '../meshes/', to: 'meshes/' }, { from: 'friction="0.6"', to: 'friction="1"' }],
  },
  {
    id: 't1',
    name: 'Booster T1',
    short: 'T1',
    emoji: '🦿',
    xmlFile: './assets/t1/T1_locomotion.xml',
    policyFile: './assets/t1/policy.onnx',
    contract: T1_CONTRACT,
    dt: 0.002,
    decim: 10,
    zHome: 0.70,
    fallZ: 0.35,
    labelH: 1.30,
    visGroups: [1],
    maxV: 1.2,
    noise: 0.01,
    wrapScene: false,
    // 官方地面 condim=1(无摩擦) -> 换成带摩擦的普通接触
    xmlPatches: [
      { from: 'condim="1"', to: 'condim="3" friction="1 0.005 0.0001"' },
    ],
  },
];

export function speciesById(id) {
  return SPECIES.find((s) => s.id === id) || SPECIES[0];
}
