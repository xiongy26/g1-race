# 🏃 双足短跑大赛(Bipedal Sprint Race)

在浏览器里让 **3 种真实双足机器人**同场进行 25 米短跑比赛:

| 物种 | 速度包线 | 模型来源 | 策略来源 |
|---|---|---|---|
| 🤖 Unitree G1(29 DoF) | 1.55 m/s | unitree_ros 官方 MJCF+网格 | [g1_deploy_mujoco](https://github.com/RoboCubPilot/g1_deploy_mujoco) ONNX |
| 🦾 众擎 SA01(12 DoF) | 1.50 m/s | [engineai_legged_gym](https://github.com/engineai-robotics/engineai_legged_gym) 官方 MJCF+网格 | 同仓库官方 `zqsa01_policy.onnx` |
| 🦿 Booster T1(12 DoF) | 1.20 m/s | [booster_gym](https://github.com/BoosterRobotics/booster_gym) 官方 MJCF+网格 | 同仓库官方 `T1.pt`(已转 ONNX) |

**收录标准:官方机器人模型文件 + 官方策略模型,两者齐备才收录;缺一不加。**
每一台都由自己的神经网络策略(ONNX @ onnxruntime-web)在 50/100Hz 闭环控制,
全部自由物理(无任何骨盆/轨道辅助)。物理由 **MuJoCo 3.14 WebAssembly** 求解,
three.js 渲染,纯前端本地运行。

### 调研记录(2026-09,为什么是这三台)

- ✅ **众擎 SA01**:`engineai_legged_gym` 同时提供官方 MJCF、STL 网格、ONNX 策略和
  `sim2sim_zqsa01.py` 部署脚本(观测契约逐条可对齐)。
- ✅ **Booster T1**:`booster_gym` 提供训练用官方 MJCF(`T1_locomotion.xml`)、部署配置
  `T1.yaml` 与 TorchScript 权重 `T1.pt`(本仓库用 CPU torch 转成 ONNX,数值误差 <1e-6)。
- ❌ **智元 A2**:官方只放了 X2 的 URDF,无 A2 模型;无任何公开 A2 运动策略 → 不加。
- ❌ **优必选 Walker S2**:无公开的模型+策略对 → 不加。
- ❌ **MicroDuck**:无公开资产 → 不加。
- ❌ **傅里叶 GR-1/N1**:模型(Wiki-GRx-Models/Menagerie)与权重(Wiki-GRx-Deploy 的
  jit 策略)都有,但策略输入契约锁在闭源 SDK(actor+encoder 双网络)或多层配置里,
  无法可靠对齐 → 暂不加,契约公开后即可按本仓库的物种接入流程补上。

## 运行方式

```bash
cd g1-race
node server.js 8137        # 或任意静态服务器: python -m http.server 8137
# 浏览器打开 http://127.0.0.1:8137/
```

> 必须通过 HTTP 访问(ES Module 与 WASM 跨域限制)。首次打开需编译 MuJoCo WASM
> 与加载三套模型资产,耐心等几秒。

## 玩法

1. 点击 **「▶ 开始比赛」**(或按空格)发枪, 3-2-1 倒计时后六台机器人起跑;
2. 右上角实时排名, 25 米终点线冲线后弹出结算面板;
3. 每次摔倒罚时 2 秒并在原地扶起; 所有选手完赛(或 90 秒超时)后比赛结束。

### 控制项

| 控件 | 说明 |
|---|---|
| 机器人数 | 2~6 台同场竞技(默认 6) |
| 阵容 | 三强混战(默认) / 全 G1 / 随机 |
| 目标速度 | 0.30~1.55 m/s, **按各物种包线等比缩放** |
| 仿真倍速 | 0.5× / 1× / 2× / 3× / 4× |
| 视角 | 跟随领跑(低机位) / 全景 / 自由(拖拽旋转、滚轮缩放) |

## 技术实现

```
浏览器
├─ MuJoCo 3.14 WASM(@mujoco/mujoco 官方绑定)
│    · 每物种一份 MjModel(官方 MJCF 原版 + 最小运行时补丁), 每台机器人独立 MjData
│    · 关节空间 PD 力矩控制(各物种自己的 kp/kd/力矩限幅/步长: G1/T1 500Hz, SA01 1kHz)
├─ ONNX Runtime Web(WASM 后端, 单线程)
│    · 每物种独立策略 @ 各自频率(G1/T1 50Hz, SA01 100Hz), 观测契约逐条对齐官方部署代码:
│      G1  : unitree_rl_gym 布局, 96×5 组优先堆叠 = 480
│      SA01: [相位, 指令×2, 关节, 关节速度, 上次动作, 角速度, 欧拉角] 47×15 帧优先 = 705
│      T1  : [重力投影, 角速度, 指令, 步态时钟, 关节, 关节速度, 上次动作] 47 单帧
│    · 动作 -> 目标角 = a·actionScale + 默认角(各物种自己的缩放/默认站姿)
├─ three.js: mjv_updateScene 管线取每个 geom 世界位姿(官方 STL 网格), 赛道/拱门/阴影
├─ 航向保持外环(纯跟踪): 航向误差 -> 各策略的 yaw 角速度指令
└─ 比赛逻辑: 倒计时发枪、实时排名、摔倒罚时扶起、结算面板
```

### 已知实现要点(踩坑记录)

- **G1 策略极限 1.55 m/s**;速度指令更高时起步必摔(详见 `test-speed-sweep.mjs`)。
- **G1 的策略顺序 ≠ mujoco 顺序**:默认角必须经 `policyToXml` 重排成 mujoco 顺序
  再喂给 PD/obs(否则观测错乱、起步即摔——本次多物种化时踩过)。
- **SA01 obs 里的欧拉角**:官方 sim2sim 用 rpy(非四元数/重力投影), 且相位时钟按
  仿真时间 / cycle_time(0.8s) 计算, 帧优先堆叠 15 帧(与 G1 的组优先不同)。
- **T1 的步态门控**:指令平滑(每周期 ±period 限幅), |cmd|≈0 时 cos/sin 与指令项
  全部清零; 关节速度归一 0.1(非 0.05)。
- **T1 官方 MJCF 地面 condim=1(无摩擦)**, 直接用会打滑, 运行时打补丁换成摩擦地面。
- **`mjvGeom.dataid` 损坏**: 本 WASM 构建对 mesh geom 返回 2 倍 dataid, 需用
  `model.geom_dataid[objid]` 还原。
- **TorchScript → ONNX**: `T1.pt` 用 `torch.jit.load + torch.onnx.export`(动态 batch)
  转换, 转换后与 torch 前向误差 <1e-6; 转换脚本思路见 RETRAIN.md。
- **rAF 不可靠**: 仿真由 4ms 定时器驱动、渲染由 rAF + 100ms 定时器兜底。

## 回归测试

```bash
node test-gaits.mjs         # 每物种 0.85×包线 25m 单测 + 三物种混合比赛(全自由物理)
node test-straightline.mjs  # G1 六道六速直线跑回归(含 1.55 极限速度)
node test-speed-sweep.mjs   # G1 速度包线扫描
```

## 文件结构

```
g1-race/
├── index.html            # UI / importmap / 启动遮罩 / 阵容选择
├── server.js             # 极简静态服务器(node server.js [port])
├── test-gaits.mjs        # 无头回归: 三物种 25m 单测 + 混合比赛
├── test-straightline.mjs # G1 直线跑回归
├── RETRAIN.md            # 提速重训指南 + 新物种接入流程
├── src/
│   ├── main.js           # 启动、多物种调度(各策略周期)、阵容、相机、HUD
│   ├── robots.js         # 🧬 物种注册表: 模型/策略来源 + 观测/PD 契约 + 调研记录
│   ├── policy.js         # 契约化观测构建(3 种布局)、ONNX 会话、PD 控制
│   ├── sim.js            # MuJoCo 加载、多物种模型编译(VFS 注入+补丁)、实例管理
│   ├── scene.js          # three.js 赛道与机器人可视化(mjv 管线)
│   └── race.js           # 比赛状态机(倒计时/排名/摔倒罚时/结算)
├── assets/
│   ├── g1_29dof.xml + meshes/ + policy.onnx     # Unitree G1(官方模型 + 第三方策略)
│   ├── sa01/zq_sa01.xml + meshes/ + policy.onnx # 众擎 SA01(官方模型 + 官方策略)
│   └── t1/T1_locomotion.xml + meshes/ + policy.onnx # Booster T1(官方模型, 权重已转 ONNX)
└── vendor/               # 本地化的 mujoco-wasm / three / onnxruntime-web
```

## 接入新物种

见 [RETRAIN.md](RETRAIN.md):只要 GitHub 上存在某机器人的官方 MJCF/URDF + 可部署
策略权重(ONNX 或 TorchScript), 按"下载资产 → 对齐观测契约 → 转权重 → 注册物种"
四步即可加入比赛, 运行时代码零改动。

## 来源与致谢

- Unitree G1:模型 [unitree_ros](https://github.com/unitreerobotics/unitree_ros) /
  [loco-lab](https://github.com/JacobEGarcia/loco-lab),策略
  [RoboCubPilot/g1_deploy_mujoco](https://github.com/RoboCubPilot/g1_deploy_mujoco)(GPL-3.0)
- 众擎 SA01:[engineai-robotics/engineai_legged_gym](https://github.com/engineai-robotics/engineai_legged_gym)
- Booster T1:[BoosterRobotics/booster_gym](https://github.com/BoosterRobotics/booster_gym)
- 物理引擎:[MuJoCo](https://mujoco.readthedocs.io)(Apache-2.0)
- 推理引擎:[onnxruntime-web](https://github.com/microsoft/onnxruntime)(MIT)
- 渲染:[three.js](https://threejs.org)(MIT)

仅供学习研究使用;真机部署请遵循各厂商与原策略作者的相关许可。
