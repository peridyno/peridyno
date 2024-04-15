import PyPeridyno as dyno


def create_ghost_particles():
    ghost = dyno.GhostParticles3f()
    num = 0
    host_pos = dyno.VectorVec3f()
    host_pos.push_back(dyno.Vector3f([1, 1, 1]))

    ghost.set_position().resize(num)
    return ghost


if __name__ == '__main__':
    scn = dyno.SceneGraph()

    ghost = create_ghost_particles()

    app = dyno.GLfwApp()
    app.set_scenegraph(scn)
    app.initialize(800, 600, True)
    app.main_loop()
