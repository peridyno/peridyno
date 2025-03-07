import PyPeridyno as dyno
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# mod = SourceModule("""
# __global__ void HitBoxes(
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
    def __init__(self):
        super().__init__()
        self.in_Topology = dyno.FInstanceDiscreteElements3f("Topology", "", dyno.FieldTypeEnum.In, self)
        self.in_Velocity = dyno.FArray3fD("Velocity", "Rigid body velocities", dyno.FieldTypeEnum.In, self)

    def in_topology(self):
        return self.in_Topology

    def in_velocity(self):
        return self.in_Velocity

    def onEvent(self, event):
        # elements = self.in_topology().get_data_ptr()
        # velocities = self.in_velocity().get_data()
        #
        # offset = elements.calculate_element_offset()
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
scn.add_node(rigid)
rigidBody = dyno.RigidBodyInfo()
rigidBody.linear_velocity = dyno.Vector3f([0.5, 0, 0])

box = dyno.BoxInfo()
box.half_length = dyno.Vector3f([0.5 * 0.065, 0.5 * 0.065, 0.5 * 0.1])
for i in range(8, 1, -1):
    for j in range(i + 1):
        rigidBody.position = dyno.Vector3f([0.5, 0.5, 0.5]) * dyno.Vector3f(
            [0.5, 1.1 - 0.13 * i, 0.12 + 0.2 * j + 0.1 * (8 - i)])
        boxAt = rigid.add_box(box, rigidBody)

sphere = dyno.SphereInfo()
sphere.radius = 0.025

rigidSphere = dyno.RigidBodyInfo()
rigidSphere.position = dyno.Vector3f([0.5, 0.75, 0.5])
sphereAt1 = rigid.add_sphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0.5, 0.95, 0.5])
sphereAt2 = rigid.add_sphere(sphere, rigidSphere)

rigidSphere.position = dyno.Vector3f([0.5, 0.65, 0.5])
sphere.radius = 0.05
sphereAt3 = rigid.add_sphere(sphere, rigidSphere)

tet = dyno.TetInfo()
tet.v = [
    dyno.Vector3f([0.5, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.2, 0.5]),
    dyno.Vector3f([0.6, 1.1, 0.5]),
    dyno.Vector3f([0.5, 1.1, 0.6]),
]
TetAt = rigid.add_tet(tet, rigidSphere)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.state_topology().connect(mapper.in_discrete_elements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(1, 1, 0))
sRender.set_alpha(0.5)
mapper.out_triangle_set().connect(sRender.in_triangle_set())
rigid.graphics_pipeline().push_module(sRender)

hit = Hit()
rigid.state_topology().connect(hit.in_topology())
rigid.state_velocity().connect(hit.in_velocity())
rigid.animation_pipeline().push_module(hit)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()
