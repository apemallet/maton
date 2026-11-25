"""Inspect a walker structure without a policy, MJX, or a display server."""

from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from maton.rendering import draw_model
from maton.walker import WalkerConfig


def preview_walker(path: Path, output: Path | None = None, *, neutral: bool = False) -> Path:
    """Save labeled side/front/top projections of the assembled walker as a PNG."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    walker = WalkerConfig.load(path)
    xml = ET.Element('mujoco', model=walker.name)
    ET.SubElement(xml, 'compiler', angle='radian')
    walker.assemble(ET.SubElement(xml, 'worldbody'), ET.SubElement(xml, 'actuator'), ET.SubElement(xml, 'sensor'))
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))  # pyright: ignore[reportAttributeAccessIssue]
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    if neutral:
        for joint in walker.motors:
            motor = joint.motor
            assert motor is not None
            index = model.joint(joint.name).qposadr[0]
            data.qpos[index] = sum(motor.target_range) / 2 if motor.neutral is None else motor.neutral
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]

    joint_count = model.njnt - 3
    legend_rows = (joint_count + 3) // 4
    footer = .19 + max(0, legend_rows - 2) * .035
    figure = Figure(figsize=(15, 6 + max(0, legend_rows - 2) * .25), dpi=120)
    figure.subplots_adjust(left=.05, right=.98, bottom=footer, top=.8, wspace=.25)
    FigureCanvasAgg(figure)
    motor_names = {joint.name for joint in walker.motors}
    # Bounding spheres safely include all geometry, regardless of pose or shape.
    low = np.min(data.geom_xpos - model.geom_rbound[:, None], axis=0)
    high = np.max(data.geom_xpos + model.geom_rbound[:, None], axis=0)
    for index, (title, plane, labels) in enumerate([
        ('Side', (0, 2), ('x (m)', 'z (m)')),
        ('Front', (1, 2), ('y (m)', 'z (m)')),
        ('Top', (0, 1), ('x (m)', 'y (m)')),
    ]):
        axes = figure.add_subplot(1, 3, index + 1)
        draw_model(model, data, axes, plane)
        anchors = {}
        for joint_id in range(3, model.njnt):
            position = tuple(np.round(data.xanchor[joint_id][list(plane)], 6))
            anchors.setdefault(position, []).append(joint_id)
        for position, joint_ids in anchors.items():
            powered = any(model.joint(j).name in motor_names for j in joint_ids)
            color = '#b34800' if powered else '#565656'
            axes.plot(*position, 'o', color=color, markersize=4)
            label = ','.join(str(j - 2) for j in joint_ids)
            axes.annotate(label, position, xytext=(6, 6), textcoords='offset points',
                          fontsize=9, color=color, bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1))
        margin = .15
        limits_low, limits_high = low[list(plane)] - margin, high[list(plane)] + margin
        if plane[1] == 2:
            limits_low[1] = min(-.05, limits_low[1])
            axes.axhline(0, color='#6e8062', linewidth=1)
        axes.set(xlim=(limits_low[0], limits_high[0]), ylim=(limits_low[1], limits_high[1]),
                 aspect='equal', xlabel=labels[0], ylabel=labels[1], title=title)
        axes.set_facecolor('#eef3f7')
        axes.grid(alpha=.2)
    pose = 'neutral motor targets' if neutral else 'zero joint angles'
    figure.suptitle(f'{walker.name} · {pose}\n'
                   f'{len(walker.bodies)} bodies · {model.njnt - 3} hinges · {model.nu} motors\n'
                   'Orange: motor joints · Gray: passive joints', fontsize=12)
    handles = [Line2D([], [], marker='o', linestyle='',
                      color='#b34800' if model.joint(j).name in motor_names else '#565656',
                      label=f'{j - 2}: {model.joint(j).name}') for j in range(3, model.njnt)]
    figure.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8)
    output = output or Path('data/walkers') / f'{walker.name}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format='png')
    return output
