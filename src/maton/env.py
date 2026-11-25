"""Configurable planar walkers with MJX dynamics and geometric range sensing."""
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from jaxtyping import Array

from maton.walker import WalkerConfig, default_walker
from maton.terrain import TerrainConfig, rectangles, sample_terrain
from maton.curriculum import CurriculumConfig, Level


@dataclass(frozen=True)
class EnvConfig:
    timestep: float = 0.01
    frame_skip: int = 2
    episode_length: int = 500
    lidar_count: int = 9
    lidar_range: float = 3.0
    reset_noise: float = 0.01
    progress_scale: float = 1.0
    progress_mode: str = "distance"
    control_cost: float = 0.001
    control_mode: str = "torque"
    upright_reward: float = 0.0
    finish_reward: float = 0.0
    friction: float = 0.9
    angle_harmonics: int = 2

    def __post_init__(self):
        if self.timestep <= 0 or self.frame_skip < 1 or self.episode_length < 1:
            raise ValueError("Time settings must be positive")
        if self.lidar_count < 1 or self.lidar_range <= 0 or self.reset_noise < 0:
            raise ValueError("Invalid sensor or reset settings")
        if self.progress_mode not in ("distance", "upright"):
            raise ValueError("progress_mode must be distance or upright")
        if self.progress_scale <= 0 or self.control_cost < 0:
            raise ValueError("Progress scale must be positive; control cost nonnegative")
        if type(self.angle_harmonics) is not int or self.angle_harmonics < 1:
            raise ValueError("angle_harmonics must be a positive integer")
        if not 0 < self.friction < 10:
            raise ValueError("friction must be positive and below 10")
        if self.finish_reward < 0:
            raise ValueError("finish_reward must be nonnegative")
        if self.control_mode not in ("torque", "position"):
            raise ValueError("control_mode must be torque or position")
        if self.upright_reward < 0:
            raise ValueError("upright_reward must be nonnegative")


class State(eqx.Module):
    data: mjx.Data
    terrain: Array
    steps: Array
    reward: Array
    terminated: Array
    truncated: Array
    action: Array
    level: Array
    success: Array

    @property
    def done(self):
        return self.terminated | self.truncated


def _xml(config: EnvConfig, terrain: TerrainConfig, walker: WalkerConfig, rect: np.ndarray | None = None, bottom: float = -1.0) -> str:
    floor = ''.join(f'<geom name="ground{i}" type="box" pos="0 0 -0.5" size="0.1 0.6 0.5" contype="1" conaffinity="2" rgba="0.5 0.6 0.5 1"/>'
                    for i in range(1 + terrain.slots * 3))
    if rect is not None:
        floor = ''.join(f'<geom name="ground{i}" type="box" pos="{(left+right)/2} 0 {(top+bottom)/2}" '
                        f'size="{(right-left)/2} 0.6 {(top-bottom)/2}" contype="1" conaffinity="2" rgba="0.5 0.6 0.5 1"/>'
                        for i, (left, right, top) in enumerate(rect))
    xml = ET.fromstring(f'''<mujoco model="{walker.name}">
    <compiler angle="radian"/>
    <option timestep="{config.timestep}" gravity="0 0 -9.81" solver="CG" iterations="6" ls_iterations="4" integrator="Euler"/>
    <default><geom friction="{config.friction} 0.01 0.001" contype="2" conaffinity="1" condim="3" solref="0.02 1"/></default>
    <visual><global offwidth="960" offheight="540"/></visual>
    <worldbody><light pos="0 -3 6"/>{floor}</worldbody>
    </mujoco>''')
    world = xml.find("worldbody")
    assert world is not None
    walker.assemble(world, ET.SubElement(xml, "actuator"), ET.SubElement(xml, "sensor"))
    return ET.tostring(xml, encoding="unicode")


class WalkerEnv:
    def __init__(self, config: EnvConfig = EnvConfig(), terrain: TerrainConfig = TerrainConfig(),
                 curriculum: CurriculumConfig | None = None, walker: WalkerConfig | None = None):
        self.config = config
        self.walker = walker or default_walker()
        self.curriculum = curriculum or CurriculumConfig((Level("course", 1, terrain),))
        self.terrain_config = terrain = self.curriculum.levels[0].terrain
        self._floor_bottom = -max(1.0, max(x.terrain.gap_height for x in self.curriculum.levels) + 0.5)
        self.model = mujoco.MjModel.from_xml_string(_xml(config, terrain, self.walker))  # pyright: ignore[reportAttributeAccessIssue]
        self.action_size = self.model.nu
        self.observation_size = 4 * (self.model.nq - 2) * config.angle_harmonics + 2 + self.model.nsensordata + config.lidar_count
        joint_ids = self.model.actuator_trnid[:, 0]
        self._motor_qpos = jnp.asarray(self.model.jnt_qposadr[joint_ids])
        self._motor_qvel = jnp.asarray(self.model.jnt_dofadr[joint_ids])
        motors = [joint.motor for joint in self.walker.motors]
        self._target_low = jnp.array([m.target_range[0] for m in motors if m])
        self._target_high = jnp.array([m.target_range[1] for m in motors if m])
        self._neutral = jnp.array([sum(m.target_range) / 2 if m.neutral is None else m.neutral for m in motors if m])
        self._kp = jnp.array([m.kp for m in motors if m])
        self._kd = jnp.array([m.kd for m in motors if m])
        self._torque_limit = jnp.array([m.torque_limit for m in motors if m])
        self.mjx_model = mjx.put_model(self.model)
        self._data = mjx.make_data(self.mjx_model)
        self._ground_count = 1 + 3 * terrain.slots

    def _model(self, terrain: Array):
        rect = rectangles(terrain, self.terrain_config)
        pos = jnp.stack(((rect[:, 0] + rect[:, 1]) / 2, jnp.zeros(len(rect)), (rect[:, 2] + self._floor_bottom) / 2), -1)
        size = jnp.stack(((rect[:, 1] - rect[:, 0]) / 2, jnp.full(len(rect), 0.6), (rect[:, 2] - self._floor_bottom) / 2), -1)
        return self.mjx_model.replace(geom_pos=self.mjx_model.geom_pos.at[:self._ground_count].set(pos),
                                      geom_size=self.mjx_model.geom_size.at[:self._ground_count].set(size),
                                      geom_rbound=self.mjx_model.geom_rbound.at[:self._ground_count].set(jnp.linalg.norm(size, axis=-1)))

    def reset(self, key: Array, level=0) -> State:
        terrain_key, pose_key = jax.random.split(key)
        terrain = jnp.stack([sample_terrain(terrain_key, x.terrain) for x in self.curriculum.levels])[level]
        pose = jax.random.uniform(pose_key, (self.model.nq,), minval=-self.config.reset_noise, maxval=self.config.reset_noise)
        data = self._data.replace(qpos=pose, qvel=jnp.zeros(self.model.nv), ctrl=jnp.zeros(self.action_size))
        data = mjx.forward(self._model(terrain), data)
        return State(data, terrain, jnp.array(0), jnp.float32(0), jnp.array(False), jnp.array(False),
                     jnp.zeros(self.action_size), jnp.asarray(level), jnp.array(False))

    def motor_control(self, data: mjx.Data, action: Array) -> Array:
        """Map normalized commands to normalized motor torque at each physics step."""
        action = jnp.clip(jnp.nan_to_num(action), -1.0, 1.0)
        if self.config.control_mode == "torque":
            return action
        target = self._neutral + action * jnp.where(action >= 0, self._target_high - self._neutral,
                                                    self._neutral - self._target_low)
        torque = self._kp * (target - data.qpos[self._motor_qpos]) - self._kd * data.qvel[self._motor_qvel]
        return jnp.clip(torque / self._torque_limit, -1.0, 1.0)

    def posture_score(self, data: mjx.Data) -> Array:
        height = jnp.clip((self.walker.spawn_height + data.qpos[1] - 0.4 * self.walker.spawn_height)
                          / (0.4 * self.walker.spawn_height), 0.0, 1.0)
        upright = jnp.clip(jnp.cos(data.qpos[2]), 0.0, 1.0)
        return height * upright

    def potential(self, data: mjx.Data) -> Array:
        """Progress field, optionally weighted by recoverable upright posture."""
        value = self.config.progress_scale * jnp.minimum(data.qpos[0], self.terrain_config.length)
        if self.config.progress_mode == "upright":
            value *= self.posture_score(data)
        return value

    def step(self, state: State, action: Array) -> State:
        action = jnp.clip(jnp.nan_to_num(action), -1.0, 1.0)
        model = self._model(state.terrain)
        def physics_step(_, data):
            ctrl = self.motor_control(data, action)
            return mjx.step(model, data.replace(ctrl=ctrl))
        data = jax.lax.fori_loop(0, self.config.frame_skip, physics_step, state.data)
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        # The progress field saturates at the goal: running off the course earns no extra reward.
        reward = self.potential(data) - self.potential(state.data)
        reward -= self.config.control_cost * jnp.mean(data.ctrl ** 2)
        if self.config.upright_reward:
            reward += self.config.upright_reward * self.config.timestep * self.config.frame_skip * self.posture_score(data)
        reward = jnp.where(finite, reward, -1.0)
        # Preserve a finite terminal observation so a failed simulation cannot poison replay.
        data = jax.lax.cond(finite, lambda: data, lambda: state.data)
        fallen = (data.qpos[1] + self.walker.spawn_height < 0.5 * self.walker.spawn_height) | (jnp.cos(data.qpos[2]) < jnp.cos(1.4))
        finished = data.qpos[0] >= self.terrain_config.length
        success = finished & ~fallen & finite
        reward += jnp.where(success, self.config.finish_reward, 0.0)
        terminated = success | ~finite | (data.qpos[0] < -1.5)
        steps = state.steps + 1
        return State(data, state.terrain, steps, reward, terminated,
                     (steps >= self.config.episode_length) & ~terminated, action,
                     state.level, success)

    def lidar(self, state: State) -> Array:
        """Exact planar ray intersections with the same boxes used by physics."""
        rect = rectangles(state.terrain, self.terrain_config)
        low = jnp.stack((rect[:, 0], jnp.full(len(rect), self._floor_bottom)), -1)
        high = jnp.stack((rect[:, 1], rect[:, 2]), -1)
        angle = jnp.linspace(-jnp.pi / 2, jnp.pi / 12, self.config.lidar_count) - state.data.qpos[2]
        direction = jnp.stack((jnp.cos(angle), jnp.sin(angle)), -1)
        origin = jnp.stack((state.data.qpos[0], self.walker.spawn_height + state.data.qpos[1]))
        direction = jnp.where(jnp.abs(direction) < 1e-6, 1e-6, direction)
        t1 = (low[None, :, :] - origin) / direction[:, None, :]
        t2 = (high[None, :, :] - origin) / direction[:, None, :]
        near = jnp.maximum(jnp.min(jnp.stack((t1, t2)), axis=0).max(-1), 0)
        far = jnp.max(jnp.stack((t1, t2)), axis=0).min(-1)
        distances = jnp.where(far >= near, near, self.config.lidar_range)
        return jnp.clip(distances.min(-1), 0, self.config.lidar_range) / self.config.lidar_range

    def fourier(self, angle: Array) -> Array:
        """sin/cos at harmonics 1..K of each angle: smooth, wrap-free, [..., 2K]."""
        phase = angle[..., None] * jnp.arange(1, self.config.angle_harmonics + 1)
        return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), -1)

    def observation(self, state: State) -> Array:
        """Absolute pitch and joint angles (parent-relative) as Fourier features; angular
        rates as Fourier features of the angle change per action; world-frame root velocity,
        foot contacts, lidar. No height, course position, or previous action."""
        qpos, qvel = state.data.qpos, state.data.qvel
        dt = self.config.timestep * self.config.frame_skip
        angles = self.fourier(qpos[2:]).ravel()
        steps = self.fourier(jnp.clip(qvel[2:], -20, 20) * dt).ravel()
        observation = jnp.concatenate((angles, steps, jnp.clip(qvel[:2], -20, 20) / 5.0,
                                       jnp.tanh(state.data.sensordata / 100.0), self.lidar(state)))
        return observation.astype(jnp.float32)

    def cpu_model(self, state: State) -> mujoco.MjModel:  # pyright: ignore[reportAttributeAccessIssue]
        """Build the corresponding CPU model for saved-episode inspection."""
        rect = np.asarray(rectangles(state.terrain, self.terrain_config))
        return mujoco.MjModel.from_xml_string(_xml(self.config, self.terrain_config, self.walker, rect, self._floor_bottom))  # pyright: ignore[reportAttributeAccessIssue]
