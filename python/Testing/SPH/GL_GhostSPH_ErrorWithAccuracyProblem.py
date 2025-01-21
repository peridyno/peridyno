import PyPeridyno as dyno
import numpy as np

def float_range(start, stop, step):
    x = start
    while x <= stop:
        yield x
        x += step

def createFluidParticles():

    fluid = dyno.ParticleFluid3f()

    host_pos = []
    host_vel = []

    lowx = -0.1
    lowy = 0
    lowz = -0.1

    highx = 0.1
    highy = 0.1
    highz = 0.1

    s = 0.005
    m_iExt = 0

    omega = 1.0
    half_s = -s / 2.0

    num = 0

    x = lowx
    y = lowy
    z = lowz

    for x in float_range(lowx, highx, s):
        for y in float_range(lowy, highy, s):
            for z in float_range(lowz, highz, s):
                p = dyno.Vector3f([x,y,z])
                host_pos.append(p)
                host_vel.append(dyno.Vector3f([0,0,0]))

    fluid.state_position().assign(host_pos)
    fluid.state_velocity().assign(host_vel)

    host_pos.clear()
    host_vel.clear()
    return fluid

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

                host_pos.append(dyno.Vector3f([x, y, z]))
                host_vel.append(dyno.Vector3f([0, 0, 0]))
                host_force.append(dyno.Vector3f([0, 0, 0]))
                host_normal.append(dyno.Vector3f([0, 1, 0]))
                host_attribute.append(attri)

    ghost.state_position().resize(num)
    ghost.state_velocity().resize(num)

    ghost.state_normal().resize(num)
    ghost.state_attribute().resize(num)

    ghost.state_position().assign(host_pos)
    ghost.state_velocity().assign(host_vel)
    ghost.state_force().assign(host_force)
    ghost.state_normal().assign(host_normal)
    ghost.state_attribute().assgin(host_attribute)

    host_pos.clear()
    host_vel.clear()
    host_force.clear()
    host_normal.clear()
    host_attribute.clear()
    return ghost


scn = dyno.SceneGraph()
scn.set_upper_bound(dyno.Vector3f([0.5,1,0.5]))
scn.set_lower_bound(dyno.Vector3f([-0.5,0,-0.5]))

cubeBoundary = dyno.CubeModel3f()
scn.add_node(cubeBoundary)
cubeBoundary.var_location().set_value(dyno.Vector3f([0,0.5,0]))
cubeBoundary.var_length().set_value(dyno.Vector3f([1,1,1]))

cube2vol = dyno.BasicShapeToVolume3f()
scn.add_node(cube2vol)
cube2vol.var_grid_spacing().set_value(0.02)
cube2vol.var_inerted().set_value(True)
cubeBoundary.connect(cube2vol.import_shape())

boundary = dyno.VolumeBoundary3f()
scn.add_node(boundary)
cube2vol.connect(boundary.import_volumes())

fluid = createFluidParticles()
scn.add_node(fluid)
ghost = createGhostParticles()
scn.add_node(ghost)

incompressibleFluid = dyno.GhostFluid3f()
scn.add_node(incompressibleFluid)
incompressibleFluid.set_dt(0.001)
fluid.connect(incompressibleFluid.import_initial_states())
ghost.connect(incompressibleFluid.import_boundary_particles())

incompressibleFluid.connect(boundary.import_particle_systems())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
