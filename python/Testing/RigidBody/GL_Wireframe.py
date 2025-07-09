import PyPeridyno as dyno

M_PI = 3.14159265358979323846

scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)

rigidBody = dyno.RigidBodyInfo()
rigidBody.linearVelocity = dyno.Vector3f([1,0,0])
box = dyno.BoxInfo()
box.halfLength = dyno.Vector3f([0.5,0.5,0.5]) * dyno.Vector3f([0.065,0.065,0.1])

for i in range(8, 1, -1):
    for j in range(i + 1):
        rigidBody.position= dyno.Vector3f([0.5, 0.5, 0.5]) * dyno.Vector3f(
            [0.5, 1.1 - 0.13 * i, 0.12 + 0.21 * j + 0.1 * (8 - i)])
        rigid.addBox(box, rigidBody)

sphere = dyno.SphereInfo()
sphere.center = dyno.Vector3f([0.5, 0.75,0.5])
sphere.radius = 0.025

rigidSphere = dyno.RigidBodyInfo()
rigid.addSphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0.5, 0.95, 0.5])
sphere.radius = 0.025
rigid.addSphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0.5, 0.65, 0.5])
sphere.radius = 0.05
rigid.addSphere(sphere, rigidSphere)

tet = dyno.TetInfo()
rigidTet = dyno.RigidBodyInfo()
tet.v = [
    dyno.Vector3f([0.5, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.2, 0.5]),
    dyno.Vector3f([0.6, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.1, 0.6]),
]
rigidSphere.position = dyno.Vector3f([0,0,0])
rigid.addTet(tet, rigidTet)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.stateTopology().connect(mapper.inDiscreteElements())
rigid.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(1,1,0))
sRender.setAlpha(1.0)
mapper.outTriangleSet().connect(sRender.inTriangleSet())
rigid.graphicsPipeline().pushModule(sRender)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()
