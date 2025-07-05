import PyPeridyno as dyno

# class ParametricModel(dyno.Node):
#     def __init__(self):
#         super().__init__()
#         self.setForceUpdate(False)
#         self.setAutoHidden(True)
#         self.varScale().setRange(0.0001,1000)
#
# class BasicShape(ParametricModel):
#     def __init__(self):
#         super().__init__()
#
#     def getNodeType(self):
#         return "Basic Shapes"

class PlaneModel(dyno.BasicShape3f):
    def __init__(self):
        dyno.BasicShape3f.__init__(self)
        self.var_LengthX = dyno.FVarf(1, "LengthX", "length X", dyno.FieldTypeEnum.Param, self)
        self.varLengthX().setRange(0.01,100)


    def caption(self):
        return "Plane"

    def getShapeType(self):
        return 0

    @property
    def varLengthX(self):
        return self.var_LengthX

    def boundingBox(self):
        center = self.varLocation().getValue()
        rot = self.varRotation().getValue()
        scale = self.varScale().getValue()

        print(center)

        q = self.computeQuaternion()


scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([2,2,2]))
scn.setLowerBound(dyno.Vector3f([-2,-2,-2]))
plane = PlaneModel()
plane.boundingBox()
scn.addNode(plane)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()

