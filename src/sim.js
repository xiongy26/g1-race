// MuJoCo WASM 加载、MEMFS 资产注入、场景组装与多机器人实例管理。

export const LANE_WIDTH = 1.35;
export const PELVIS_HOME_Z = 0.793;
export const FALL_Z = 0.45;

export class Sim {
  // assets: { xml: string, meshes: Map<name, Uint8Array>, sceneXml: string }
  static async load(mujocoMod, assets, log) {
    const sim = new Sim();
    sim.mj = mujocoMod;

    log('注入 MJCF 与网格到模型虚拟文件系统 ...');
    const vfs = new mujocoMod.MjVFS();
    vfs.addBuffer('g1_29dof.xml', new TextEncoder().encode(assets.xml));
    for (const [name, bytes] of assets.meshes) {
      vfs.addBuffer('meshes/' + name, bytes);
    }

    log('编译 MuJoCo 模型(29 DoF + 36 网格)...');
    sim.model = mujocoMod.MjModel.from_xml_string(assets.sceneXml, vfs);
    if (!sim.model) throw new Error('MjModel 编译失败');
    sim.nq = sim.model.nq; sim.nu = sim.model.nu; sim.nv = sim.model.nv;
    log(`模型就绪: nq=${sim.nq} nv=${sim.nv} nu=${sim.nu} nbody=${sim.model.nbody} ngeom=${sim.model.ngeom}`);
    return sim;
  }

  // 创建一台机器人(独立 MjData, 共享 MjModel)。
  // 注意: 不要缓存 data.qpos/ctrl 的包装对象, 每次经 data.<field> 现取。
  addRobot() {
    const data = new this.mj.MjData(this.model);
    return { data };
  }

  resetRobot(robot, laneY, rng, noiseScale = 0.02, startX = 0) {
    const { data } = robot;
    const qpos = data.qpos, qvel = data.qvel;
    this.mj.mj_resetData(this.model, data);
    qpos[0] = startX;   // 起点线(扶起时为当前位置)
    qpos[1] = laneY;    // 赛道
    qpos[2] = PELVIS_HOME_Z;
    qpos[3] = 1; qpos[4] = 0; qpos[5] = 0; qpos[6] = 0;
    // 关节初始小扰动 -> 各机器人轨迹分岔, 比赛才有差别
    for (let i = 7; i < this.nq; i++) {
      qpos[i] = (rng() * 2 - 1) * noiseScale;
    }
    for (let i = 0; i < this.nv; i++) qvel[i] = 0;
    this.mj.mj_forward(this.model, data);
  }

  // robot 为主机器人对象 { sim: { data }, ... }, 数据经 robot.sim.data 现取
  step(robot) {
    this.mj.mj_step(this.model, robot.sim.data);
  }}

// 比赛场景: 在机器人 MJCF 外包一层地面与求解器配置
export function buildSceneXml() {
  return `<mujoco model="g1_race">
  <include file="g1_29dof.xml"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0" rgba="0.98 0.98 0.98 1" condim="3" friction="1 0.005 0.0001"/>
  </worldbody>
</mujoco>`;
}

// 可复现的伪随机数(mulberry32)
export function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
