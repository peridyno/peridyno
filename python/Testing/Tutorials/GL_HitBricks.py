import PyPeridyno as dyno
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# mod = SourceModule("""
# Global void HitBoxes(
# 	DArray<TOrientedBox3D<float>> box,
# 	DArray<Vec3f> velocites,
# 	TRay3D<float> ray,
# 	int boxIndex)
# {
# 	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
# 	if (tId >= box.size()) return;
#
# 	TSegment3D<float> seg;
#
# 	if (ray.intersect(box[tId], seg) > 0)
# 	{
# 		velocites[tId + boxIndex] += 10.0f * ray.direction;
# 	};
# }
# """)

class Hit(dyno.MouseInputModule):
    def Init(self):
        super().Init()
        self.in_Topology = dyno.FInstanceDiscreteElements3f("Topology", "", dyno.FieldTypeEnum.In, self)
        self.in_Velocity = dyno.FArray3fD("Velocity", "Rigid body velocities", dyno.FieldTypeEnum.In, self)

    def inTopology(self):
        return self.in_Topology

    def inVelocity(self):
        return self.in_Velocity

    def onEvent(self, event):
        # elements = self.inTopology().getDataPtr()
        # velocities = self.inVelocity().getData()
        #
        # offset = elements.calculateElementOffset()
        # boxInGlobal = dyno.DArrayTOrientedBox3D()
        #
        # elements.requestBoxInGlobal(boxInGlobal)
        print("Python onEvent")
        print(event.actionType)
        if event.actionType == dyno.PActionType.AT_UNKOWN:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if event.actionType == dyno.PActionType.AT_RELEASE:
            print("111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")

        if event.actionType == dyno.PActionType.AT_PRESS:
            print("222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222")

        if event.actionType == dyno.PActionType.AT_REPEAT:
            print("333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333")

        # boxInGlobal.clear()


scn = dyno.SceneGraph()

rigid = dyno.RigidBodySystem3f()
scn.addNode(rigid)
rigidBody = dyno.RigidBodyInfo()
rigidBody.linearVelocity = dyno.Vector3f([0.5, 0, 0])

box = dyno.BoxInfo()
box.halfLength = dyno.Vector3f([0.5 * 0.065, 0.5 * 0.065, 0.5 * 0.1])
for i in range(8, 1, -1):
    for j in range(i + 1):
        rigidBody.position = dyno.Vector3f([0.5, 0.5, 0.5]) * dyno.Vector3f(
            [0.5, 1.1 - 0.13 * i, 0.12 + 0.2 * j + 0.1 * (8 - i)])
        boxAt = rigid.addBox(box, rigidBody)

sphere = dyno.SphereInfo()
sphere.radius = 0.025

rigidSphere = dyno.RigidBodyInfo()
rigidSphere.position = dyno.Vector3f([0.5, 0.75, 0.5])
sphereAt1 = rigid.addSphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0.5, 0.95, 0.5])
sphereAt2 = rigid.addSphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0.5, 0.65, 0.5])
sphere.radius = 0.05
sphereAt3 = rigid.addSphere(sphere, rigidSphere)

tet = dyno.TetInfo()
tet.v = [
    dyno.Vector3f([0.5, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.2, 0.5]),
    dyno.Vector3f([0.6, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.1, 0.6]),
]
TetAt = rigid.addTet(tet, rigidSphere)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.stateTopology().connect(mapper.inDiscreteElements())
rigid.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(1, 1, 0))
sRender.setAlpha(0.5)
mapper.outTriangleSet().connect(sRender.inTriangleSet())
rigid.graphicsPipeline().pushModule(sRender)

hit = Hit()
rigid.stateTopology().connect(hit.inTopology())
rigid.stateVelocity().connect(hit.inVelocity())
rigid.animationPipeline().pushModule(hit)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()
