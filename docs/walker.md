# Walker structure

Each job sets `walker = "path/to/walker.toml"`, relative to that job.
Examples live in [configs/walkers](../configs/walkers):

- `basic.toml`: original four-motor biped with spherical feet.
- `biped.toml`: capsule feet and straight neutral position targets; used by the walking job.
- `quadruped.toml`: four legs, eight motors; select it by changing the job's `walker` path.

Preview a structure without training or a display server:

```sh
just view-walker                              # default biped
just view-walker configs/walkers/quadruped.toml
just view-walker configs/walkers/basic.toml --neutral --output data/basic.png
```

Saves `data/walkers/<name>.png` with side/front/top views, numbered joints, and a
joint-name key. Default pose has zero joint angles; `--neutral` uses motor neutral
targets, leaving passive joints at zero. CLI: `uv run maton view-walker FILE [--neutral] [--output PNG]`.

Structures are assembled into MuJoCo bodies, joints, motors, and touch sensors.
Run snapshots embed the full resolved structure, so evaluation needs no source TOML.
Changing structure requires a new policy with matching input/output sizes.

```toml
name = "hopper"
spawn_height = 0.8

[[bodies]]
name = "torso"                 # exactly one body without a parent
[[bodies.geoms]]
type = "box"
size = [0.12, 0.1, 0.15]       # box half-extents
mass = 2.0

[[bodies]]
name = "leg"
parent = "torso"
pos = [0, 0, -0.15]           # parent-relative attachment point
[bodies.joint]                # omit for a fixed link
name = "hip"
range = [-1.2, 1.2]           # radians; must contain initial angle zero
[bodies.joint.motor]          # omit for a passive hinge
# Actions are normalized to [-1, 1] in both control modes.
torque_limit = 20.0
kp = 30.0
kd = 2.0
target_range = [-1.0, 1.0]     # inside physical range
neutral = 0.0                 # defaults to target midpoint
[[bodies.geoms]]
type = "capsule"
fromto = [0, 0, 0, 0, 0, -0.5]
size = [0.05]                 # radius
mass = 0.8
contact = true                # adds one touch sensor
```

- Units: metres, kilograms, radians. Names use letters, digits, underscores;
  `root_` is reserved. Unknown fields are rejected.
- Root: x/z slides and pitch hinge, placed at `spawn_height`; root `pos` is zero.
  Height-based posture and success thresholds scale with `spawn_height`.
- Layout: unique body names and a connected, acyclic parent tree. Declaration order
  need not put parents first. Each body has at least one mass-bearing geom.
- Joints: one optional y-axis hinge per body; `damping=0.5`, `armature=0.02`.
  Fixed links add no coordinates. Passive hinges are observed but receive no action.
- Geometry: `sphere` (one radius), `capsule` (radius plus `fromto` endpoints),
  `box` (three half-extents). `pos=[0,0,0]`, `rgba=[0.2,0.35,0.8,1]`,
  `contact=false` by default. Capsule endpoints define its placement.
- Motors: at least one required. Gains and torque limits are per joint.
  `[env].control_mode` chooses torque or position control for the whole walker.
- Ordering: depth-first body traversal, preserving sibling declaration order.
  Actions follow motor joints; contact observations follow marked geoms.
  Input/output sizes derive from the assembled model, including passive joints.
- Physics remains planar. Walker geoms collide with terrain, without self-collision.
