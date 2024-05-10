import os

import PyPeridyno as dyno


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


scn = dyno.SceneGraph()

scn.set_total_time(3)
scn.set_gravity(dyno.Vector3f([0, -9.8, 0]))
scn.set_lower_bound(dyno.Vector3f([-0.5, 0, -4]))
scn.set_upper_bound(dyno.Vector3f([0.5, 1, 4]))

velocity = dyno.Vector3f([0, 0, 6])
color = dyno.Color(1, 1, 1)

LocationBody = dyno.Vector3f([0, 0.01, -1])
anglurVel = dyno.Vector3f([100, 0, 0])
scale = dyno.Vector3f([0.4, 0.4, 0.4])

ObjJeep = dyno.ObjMesh3f()

scn.add_node(ObjJeep)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
