import os

import PyPeridyno as dyno

scn = dyno.SceneGraph()


def filePath(str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../../data/" + str
    file_path = os.path.join(script_dir, relative_path)
    if os.path.isfile(file_path):
        print(file_path)
        return file_path
    else:
        print(f"File not found: {file_path}")
        return -1


elastoplasticBody = dyno.ElastoplasticBody3f()
elastoplasticBody.set_visible(False)
elastoplasticBody.load_particles(dyno.Vector3f([-1.1, -1.1, -1.1]), dyno.Vector3f([1.15, 1.15, 1.15]), 0.1)
elastoplasticBody.scale(0.05)
elastoplasticBody.translate(dyno.Vector3f([0.3, 0.2, 0.5]))

surfaceMeshLoader = dyno.SurfaceMeshLoader3f()

scn.add_node(elastoplasticBody)
scn.add_node(surfaceMeshLoader)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
