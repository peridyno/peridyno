import PyPeridyno as dyno
import numpy as np


def createGhostParticles():
    ghost = dyno.GhostParticles3f()
    host_pos = dyno.VectorVec3f()
    host_vel = dyno.VectorVec3f()
    host_force = dyno.VectorVec3f()
    host_normal = dyno.VectorVec3f()
    host_attribute = dyno.VectorAttribute()

    low = dyno.Vector3f([-0.2, -0.015, -0.2])
    high = dyno.Vector3f([0.2, -0.005, 0.2])

    lowx = -0.2
    lowy = -0.015
    lowz = -0.2

    highx = 0.2
    highy = -0.005
    highz = 0.2

    s = 0.005
    m_iExt = 0

    omega = 1.0
    half_s = -s / 2.0

    num = 0
    for x in np.arange(lowx - m_iExt * s, highx + m_iExt * s + s, s):
        for y in np.arange(lowy - m_iExt * s, highy + m_iExt * s, s):
            for z in np.arange(lowz - m_iExt * s, highz + m_iExt * s, s):
                print(x, y, z)
                attri = dyno.Attribute()
                attri.set_fluid()
                attri.set_dynamic()

                # host_pos.append(dyno.Vector3f([x, y, z]))
                # host_vel.append(dyno.Vector3f([0, 0, 0]))
                # host_force.append(dyno.Vector3f([0, 0, 0]))
                # host_normal.append(dyno.Vector3f([0, 1, 0]))
                # host_attribute.append(attri)

    ghost.state_position().resize(num)
    ghost.state_velocity().resize(num)
    ghost.state_force().resize(num)

    ghost.state_normal().resize(num)
    ghost.state_attribute().resize(num)

    # ghost.state_position().assign(host_pos)
    # ghost.state_velocity().assign(host_vel)
    # ghost.state_force().assign(host_force)
    # ghost.state_normal().assign(host_normal)
    # ghost.state_attribute().assgin(host_attribute)

    host_pos.clear()
    host_vel.clear()
    host_force.clear()
    host_normal.clear()
    host_attribute.clear()
    return ghost


scene = dyno.SceneGraph()
ghost = createGhostParticles()
