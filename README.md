# 🏃 G1 短跑大赛(G1 Sprint Race)

在浏览器里让多台 **Unitree G1(29 DoF)** 人形机器人进行 25 米短跑比赛。
物理由 **MuJoCo 3.14 WebAssembly** 实时求解,每台机器人由一个**真实的强化学习步态策略(ONNX 神经网络)**闭环控制,three.js 渲染,纯前端本地运行,无需任何服务器后端或 GPU。

## 运行方式

```bash
cd g1-race
node server.js 8137        # 或任意静态服务器: python -m http.server 8137
# 浏览器打开 http://127.0.0.1:8137/
```

> 必须通过 HTTP 访问(不能直接双击 index.html),因为 ES Module 与 WASM 资源有跨域限制。
> 首次打开需编译约 10MB 的 MuJoCo WASM 与加载资产(全部为本地文件),耐心等几秒。

## 玩法

1. 点击 **「▶ 开始比赛」**(或按空格)发枪,3-2-1 倒计时后四台机器人起跑;
2. 右上角实时排名,25 米终点线(方格旗 + 拱门)冲线后弹出结算面板;
3. 每次摔倒罚时 2 秒并在原地扶起;所有选手完赛(或 90 秒超时)后比赛结束。

### 控制项

| 控件 | 说明 |
|---|---|
| 机器人数 | 2~6 台同场竞技 |
| 目标速度 | 0.30~1.00 m/s(策略训练范围),每台附带随机 ±0.1 m/s 配速差 |
| 仿真倍速 | 0.5× / 1× / 2× |
| 视角 | 跟随领跑 / 全景 / 自由(拖拽旋转、滚轮缩放) |

## 技术实现

```
浏览器
├─ MuJoCo 3.14 WASM(@mujoco/mujoco 官方绑定)
│    · 每台机器人独立 MjData,共享同一 MjModel(29 DoF, dt=0.002)
│    · 500 Hz 关节空间 PD 力矩控制(ctrl = kp*(q*-q) - kd*q̇)
├─ ONNX Runtime Web(WASM 后端, 单线程)
│    · 速度策略: 输入 obs[1×480], 输出 action[1×29], 50 Hz 闭环
│    · obs(96 维) = [机体角速度×0.2, 重力投影, 速度指令, 关节角-默认, 关节角速度×0.05, 上次动作]
│    · 5 帧堆叠按"分量组优先"打包成 480 维; 动作→目标角 = 0.25×a + 默认站姿
├─ three.js
│    · 机器人网格经 mjv_updateScene 管线取每个 geom 的最终世界位姿
│    · 程序化赛道纹理(道次、起点线、方格旗终点、里程标)、终点拱门、阴影
└─ 比赛逻辑: 倒计时发枪、实时排名、摔倒判定(z<0.45)与扶起、罚时、结算面板
```

### 已知实现要点(踩坑记录)

- **`mjvGeom.dataid` 损坏**:本 WASM 构建对 mesh geom 返回 2 倍的错误 dataid,必须用
  `model.geom_dataid[mjvGeom.objid]` 还原真实 mesh id,否则 STL 错配、机器人"炸开"。
- **网格加载**:官方绑定需用 `MjVFS.addBuffer('meshes/xxx.STL', bytes)` +
  `MjModel.from_xml_string(xml, vfs)`,meshdir 才能正确解析。
- **embind 数组**:MjData 的 `qpos/qvel/ctrl` 每次经 `data.<field>` 现取,不要缓存包装对象。
- **rAF 不可靠**:部分嵌入环境会冻结 requestAnimationFrame,仿真由 4ms 定时器驱动、
  渲染由 rAF + 100ms 定时器兜底,保证任何环境下都能推进。

## 文件结构

```
g1-race/
├── index.html            # UI / importmap / 启动遮罩
├── server.js             # 极简静态服务器(node server.js [port])
├── src/
│   ├── main.js           # 启动流程、主循环(仿真/渲染解耦)、相机、HUD
│   ├── policy.js         # 策略观测构建、ONNX 会话封装、PD 控制
│   ├── sim.js            # MuJoCo 加载、MjVFS 注入、多机器人实例管理
│   ├── scene.js          # three.js 赛道与机器人可视化(mjv 管线)
│   └── race.js           # 比赛状态机(倒计时/排名/摔倒罚时/结算)
├── assets/
│   ├── g1_29dof.xml      # Unitree G1 29DoF MJCF
│   ├── meshes/*.STL      # 36 个网格
│   └── policy.onnx       # 预训练速度策略(1.7MB MLP)
└── vendor/               # 本地化的 mujoco-wasm / three / onnxruntime-web
```

## 来源与致谢

- 机器人模型:[JacobEGarcia/loco-lab](https://github.com/JacobEGarcia/loco-lab)(G1 29DoF MJCF + 网格),
  网格原始出处 [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros)
- 强化学习策略:`policy.onnx` 转换自开源的
  [RoboCubPilot/g1_deploy_mujoco](https://github.com/RoboCubPilot/g1_deploy_mujoco)
  (unitree_rl_gym 风格 G1 速度策略,GPL-3.0),观测/动作约定参考 loco-lab 的验证实现
- 物理引擎:[MuJoCo](https://mujoco.readthedocs.io)(Apache-2.0)官方 JS 绑定 `@mujoco/mujoco`
- 推理引擎:[onnxruntime-web](https://github.com/microsoft/onnxruntime)(MIT)
- 渲染:[three.js](https://threejs.org)(MIT)

仅供学习研究使用;真机部署请遵循 Unitree 与原策略作者的相关许可。
