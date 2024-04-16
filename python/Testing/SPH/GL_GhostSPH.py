import PyPeridyno as dyno


def create_ghost_particles():
    ghost = dyno.GhostParticles3f()
    return ghost


if __name__ == '__main__':
    scn = dyno.SceneGraph()

    ghost = create_ghost_particles()

    app = dyno.GLfwApp()
    app.set_scenegraph(scn)
    app.initialize(800, 600, True)
    app.main_loop()
