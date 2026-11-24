"""Planar four-motor biped with MJX dynamics and geometric range sensing."""
from dataclasses import dataclass
import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from jaxtyping import Array

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
    control_cost: float = 0.001

    def __post_init__(self):
        if self.timestep <= 0 or self.frame_skip < 1 or self.episode_length < 1:
            raise ValueError("Time settings must be positive")
        if self.lidar_count < 1 or self.lidar_range <= 0 or self.reset_noise < 0:
            raise ValueError("Invalid sensor or reset settings")
        if self.progress_scale <= 0 or self.control_cost < 0:
            raise ValueError("Progress scale must be positive; control cost nonnegative")


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


def _xml(config: EnvConfig, terrain: TerrainConfig) -> str:
    floor = ''.join(f'<geom name="ground{i}" type="box" pos="0 0 -0.5" size="0.1 0.6 0.5" contype="1" conaffinity="2" rgba="0.5 0.6 0.5 1"/>'
                    for i in range(1 + terrain.slots * 3))
    legs = ''.join(f'''
      <body name="thigh_{side}" pos="0 {y} -0.15">
        <joint name="hip_{side}" type="hinge" axis="0 1 0" range="-80 80"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.045" mass="0.8"/>
        <body name="shin_{side}" pos="0 0 -0.35">
          <joint name="knee_{side}" type="hinge" axis="0 1 0" range="-5 125"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.035" mass="0.5"/>
          <geom name="foot_{side}" type="sphere" pos="0 0 -0.35" size="0.075" mass="0.15"/>
          <site name="touch_{side}" pos="0 0 -0.35" size="0.085"/>
        </body>
      </body>''' for side, y in [('left', -0.11), ('right', 0.11)])
    motors = ''.join(f'<motor joint="{joint}_{side}" gear="20" ctrllimited="true" ctrlrange="-1 1"/>'
                     for side in ['left', 'right'] for joint in ['hip', 'knee'])
    return f'''<mujoco model="planar_walker">
    <compiler angle="degree"/>
    <option timestep="{config.timestep}" gravity="0 0 -9.81" solver="CG" iterations="6" ls_iterations="4" integrator="Euler"/>
    <default><joint damping="0.5" armature="0.02"/><geom friction="0.9 0.01 0.001" contype="2" conaffinity="1" condim="3" solref="0.02 1"/></default>
    <visual><global offwidth="960" offheight="540"/></visual>
    <worldbody><light pos="0 -3 6"/>{floor}
    <body name="torso" pos="0 0 1.0">
      <joint name="root_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="root_z" type="slide" axis="0 0 1" damping="0"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" limited="false" damping="0.1"/>
      <geom type="capsule" fromto="0 0 -0.15 0 0 0.15" size="0.09" mass="3" rgba="0.2 0.35 0.8 1"/>
      {legs}
    </body></worldbody><actuator>{motors}</actuator>
    <sensor><touch name="left_contact" site="touch_left"/><touch name="right_contact" site="touch_right"/></sensor>
    </mujoco>'''


class WalkerEnv:
    action_size = 4

    def __init__(self, config: EnvConfig = EnvConfig(), terrain: TerrainConfig = TerrainConfig(),
                 curriculum: CurriculumConfig | None = None):
        self.config = config
        self.curriculum = curriculum or CurriculumConfig((Level("course", 1, terrain),))
        self.terrain_config = terrain = self.curriculum.levels[0].terrain
        self._floor_bottom = -max(1.0, max(x.terrain.gap_height for x in self.curriculum.levels) + 0.5)
        self.observation_size = 20 + config.lidar_count
        self.model = mujoco.MjModel.from_xml_string(_xml(config, terrain))  # pyright: ignore[reportAttributeAccessIssue]
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
        pose = jax.random.uniform(pose_key, (7,), minval=-self.config.reset_noise, maxval=self.config.reset_noise)
        data = self._data.replace(qpos=pose, qvel=jnp.zeros(7), ctrl=jnp.zeros(4))
        data = mjx.forward(self._model(terrain), data)
        return State(data, terrain, jnp.array(0), jnp.float32(0), jnp.array(False), jnp.array(False),
                     jnp.zeros(4), jnp.asarray(level), jnp.array(False))

    def step(self, state: State, action: Array) -> State:
        action = jnp.clip(jnp.nan_to_num(action), -1.0, 1.0)
        model = self._model(state.terrain)
        data = state.data.replace(ctrl=action)
        data = jax.lax.fori_loop(0, self.config.frame_skip, lambda _, d: mjx.step(model, d), data)
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        # The potential is Phi(x) = progress_scale * x; its difference pays only new progress.
        reward = self.config.progress_scale * (data.qpos[0] - state.data.qpos[0])
        reward -= self.config.control_cost * jnp.mean(action ** 2)
        reward = jnp.where(finite, reward, -1.0)
        # Preserve a finite terminal observation so a failed simulation cannot poison replay.
        data = jax.lax.cond(finite, lambda: data, lambda: state.data)
        fallen = (data.qpos[1] + 1.0 < 0.5) | (jnp.abs(data.qpos[2]) > 1.4)
        finished = data.qpos[0] >= self.terrain_config.length
        success = finished & ~fallen & finite
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
        origin = jnp.stack((state.data.qpos[0], 1.0 + state.data.qpos[1]))
        direction = jnp.where(jnp.abs(direction) < 1e-6, 1e-6, direction)
        t1 = (low[None, :, :] - origin) / direction[:, None, :]
        t2 = (high[None, :, :] - origin) / direction[:, None, :]
        near = jnp.maximum(jnp.min(jnp.stack((t1, t2)), axis=0).max(-1), 0)
        far = jnp.max(jnp.stack((t1, t2)), axis=0).min(-1)
        distances = jnp.where(far >= near, near, self.config.lidar_range)
        return jnp.clip(distances.min(-1), 0, self.config.lidar_range) / self.config.lidar_range

    def observation(self, state: State) -> Array:
        qpos, qvel = state.data.qpos, state.data.qvel
        return jnp.concatenate((jnp.array([1.0 + qpos[1], jnp.sin(qpos[2]), jnp.cos(qpos[2])]),
                                qpos[3:], jnp.clip(qvel, -20, 20),
                                jnp.tanh(state.data.sensordata / 100.0), state.action, self.lidar(state))).astype(jnp.float32)

    def cpu_model(self, state: State) -> mujoco.MjModel:  # pyright: ignore[reportAttributeAccessIssue]
        """Build the corresponding CPU model for saved-episode inspection."""
        model = copy.copy(self.model)
        updated = self._model(state.terrain)
        model.geom_pos[:] = np.asarray(updated.geom_pos)
        model.geom_size[:] = np.asarray(updated.geom_size)
        model.geom_rbound[:] = np.asarray(updated.geom_rbound)
        return model
