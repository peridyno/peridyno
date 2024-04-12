import PyPeridyno as dyno
import sys
print(sys.path)

scn = dyno.SceneGraph()

scn.set_upper_bound(dyno.Vector3f([1.5, 1, 1.5]))
scn.set_lower_bound(dyno.Vector3f([-0.5, 0, -0.5]))

cube = dyno.CubeModel3f()

app = dyno.GLApp()
app.set_scenegraph(scn)
app.initialize(800, 600, True)
app.main_loop()
