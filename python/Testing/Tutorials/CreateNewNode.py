import PyPeridyno as dyno

class VolumeTest(dyno.Node):
    def __init__(self):
        super().__init__()

        self.state_LevelSet = dyno.FInstanceLevelSet3f("LevelSet", "", dyno.FieldTypeEnum.State, self)

        self.set_auto_hidden(True)
        mapper = dyno.VolumeToTriangleSet3f()
        self.state_level_set().connect(mapper.io_volume())
        self.graphics_pipeline().push_module(mapper)

        renderer = dyno.GLSurfaceVisualModule()
        mapper.out_triangle_set().connect(renderer.in_triangle_set())
        self.graphics_pipeline().push_module(renderer)

    def get_node_type(self):
        return "Volume"

    def state_level_set(self):
        return self.state_LevelSet


scn = dyno.SceneGraph()
scn.set_upper_bound(dyno.Vector3f([2,2,2]))
scn.set_lower_bound(dyno.Vector3f([-2,-2,-2]))
test = VolumeTest()
scn.add_node(test)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.set_window_title("Empty GUI")
app.main_loop()

