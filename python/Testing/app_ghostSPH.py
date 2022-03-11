import PyPeridyno as dyno
import numpy as np


scn = dyno.SceneGraph()


def createGhostParticles():
    ghost = dyno.GhostParticles3f()

    host_pos = []
    host_vel = []
    host_force = []
    host_normal = []
    host_attribute = []



    low = dyno.Vector3f([-0.2, -0.015, -0.2])
    high = dyno.Vector3f([0.2, -0.005, 0.2])

    s = 0.005
    m_iExt = 0
    omega = 1.0
    half_s = -s / 2.0
    num = 0

    #for x in range(low.x - m_iExt * s, high.x + m_iExt * s, s):

    for x in np.arange(low[0] - m_iExt * s, high[0] + m_iExt * s, s):
        for y in np.arange(low[1] - m_iExt * s, high[1] + m_iExt * s, s):
            for z in np.arange(low[2] - m_iExt * s, high[2] + m_iExt * s, s):
                attri = dyno.Attribute()
                attri.set_fluid()
                attri.set_dynamic()

                host_pos.append(dyno.Vector3f([x, y, z]))
                host_vel.append(dyno.Vector3f([0, 0, 0]))
                host_force.append(dyno.Vector3f([0, 0, 0]))
                host_normal.append(dyno.Vector3f([0, 1, 0]))
                host_attribute.append(attri)

    ghost.state_position().set_elementCount(num)
    ghost.state_velocity().set_elementCount(num)
    ghost.state_force().set_elementCount(num)

    ghost.state_normal().set_elementCount(num)
    #ghost.state_attribute().set_elementCount(num)

    #ghost.state_position().get_dataPtr()



    return 1

if __name__ == '__main__':
    scn.set_upper_bound(dyno.Vector3f([1.5, 1, 1.5]))
    scn.set_lower_bound(dyno.Vector3f([-0.5, 0, -0.5]))

    boundary = dyno.StaticBoundary3f()
    boundary.load_cube(dyno.Vector3f([-0.5, 0, -0.5]), dyno.Vector3f([0.1, 1.0, 0.1]), 0.005, True)
    boundary.load_sdf("../../data/bowl/bowl.sdf", False)
    scn.add_node(boundary)

    print(createGhostParticles())


    app = dyno.GLApp()
    app.set_scenegraph(scn)
    app.create_window(1024, 768)
    app.main_loop()

