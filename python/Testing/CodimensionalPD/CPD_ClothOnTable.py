import PyPeridyno as dyno
import os

scn = dyno.SceneGraph()
scn.set_lower_bound(dyno.Vector3f([-1.5, 0, -1.5]))
scn.set_upper_bound(dyno.Vector3f([1.5, 3, 1.5]))

mesh = dyno.StaticTriangularMesh3f()
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../data/cloth_shell/table/table.obj"
file_path = os.path.join(script_dir, relative_path)
if os.path.isfile(file_path):
    FilePath = dyno.FilePath(file_path)
    mesh.var_file_name().set_value(FilePath)
else:
    print(f"File not found: {file_path}")
boundary = 


scn.add_node(mesh)
scn.add_node(boundary)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(800, 600, True)
app.main_loop()
