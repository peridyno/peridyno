import PyPeridyno as dyno

def create_ghost_particles():
    

if __name__=='__main__':
    scn = dyno.SceneGraph()

    app = dyno.GLfwApp()
    app.set_scenegraph(scn)
    app.initialize(800, 600, True)
    app.main_loop()