import PyPeridyno as dyno

# class ParametricModel(dyno.Node):
#     def __init__(self):
#         super().__init__()
#         self.setForceUpdate(False)
#         self.setAutoHidden(True)
#         # self.varScale().setRange(0.0001, 1000)
#
# class BasicShape(ParametricModel):
#     def __init__(self):
#         super().__init__()
#
#     def getNodeType(self):
#         return "Basic Shapes"
#
#     def getShapeType(self):
#         return dyno.BasicShapeType.UNKNOWN


class PlaneModel(dyno.BasicShape3f):
    def __init__(self):
        super().__init__()
        self.var_LengthX = dyno.FVarf(1.0, "LengthX", "length X", dyno.FieldTypeEnum.Param, self)
        self.var_LengthZ = dyno.FVarf(1,"LengthZ", "LengthZ", dyno.FieldTypeEnum.Param, self)
        self.var_SegmentX = dyno.FVarf(1, "SegmentX", "SegmentX",dyno.FieldTypeEnum.Param, self)
        self.var_SegmentZ = dyno.FVarf(1, "SegmentZ", "SegmentZ", dyno.FieldTypeEnum.Param, self)

        self.state_PolygonSet = dyno.FInstance("PolygonSet", "", dyno.FieldTypeEnum.State, self)
        self.state_TriangleSet = dyno.FInstance("TriangleSet", "", dyno.FieldTypeEnum.State, self)
        self.state_QuadSet = dyno.FInstance("QuadSet", "", dyno.FieldTypeEnum.State, self)

        self.varLengthX.setRange(0.01, 100)
        self.varLengthZ.setRange(1, 100)
        self.varSegmentX.setRange(1, 100)
        self.varSegmentZ.setRange(1, 100)

        # Initialize data structures
        self.statePolygonSet.setDataPtr(dyno.PolygonSet3f())
        self.stateTriangleSet.setDataPtr(dyno.TriangleSet3f())
        self.stateQuadSet.setDataPtr(dyno.QuadSet3f())

        # Rending
        tsRender = dyno.GLSurfaceVisualModule()
        self.stateTriangleSet.connect(tsRender.inTriangleSet())
        self.graphicsPipeline().pushModule(tsRender)

        exES = dyno.ExtractTriangleSetFromPolygonSet3f()
        self.statePolygonSet.connect(exES.inPolygonSet())
        self.graphicsPipeline().pushModule(exES)

        esRender = dyno.GLWireframeVisualModule()
        esRender.varBaseColor().setValue(dyno.Color(0,0,0))
        self.stateTriangleSet.connect(esRender.inEdgeSet())
        self.graphicsPipeline().pushModule(esRender)

        self.stateTriangleSet.promoteOuput()
        self.stateQuadSet.promoteOuput()
        self.statePolygonSet.promoteOuput()

    def caption(self):
        return "Plane"

    def getShapeType(self):
        return dyno.BasicShapeType.PLANE

    @property
    def varLengthX(self):
        return self.var_LengthX

    @property
    def varLengthZ(self):
        return self.var_LengthZ

    @property
    def varSegmentX(self):
        return self.var_SegmentX

    @property
    def varSegmentZ(self):
        return self.var_SegmentZ

    @property
    def statePolygonSet(self):
        return self.state_PolygonSet

    @property
    def stateTriangleSet(self):
        return self.state_TriangleSet

    @property
    def stateQuadSet(self):
        return self.state_QuadSet

    def reset(self):
        print("python reset")
        # self.resetStates()

    def resetStates(self):
        print("python resetStates")
        # self.varChange()


    def varChange(self):
        print("update")
        center = self.varLocation().getData()
        rot = self.varRotation().getData()
        scale = self.varScale().getData()

        segmentX = self.varSegmentX.getData()
        segmentZ = self.varSegmentZ.getData()

        lengthX = self.varLengthX.getData()
        lengthZ = self.varLengthZ.getData()

        length = dyno.Vector3f(lengthX, 1, lengthZ)
        segments = dyno.Vector3i(segmentX, 1, segmentZ)

        lengthX *= scale[0]
        lengthZ *= scale[2]

        q = self.computeQuaternion()
        q.normalize()

        vertices = []
        quads = []
        triangles = []

        dx = lengthX / segmentX
        dz = lengthZ / segmentZ

        # Lambda function to rotate a vertex
        def RV(v):
            return center + q.rotate(v)

        numOfPolygon = segments[0] * segments[2]
        counter2 = [4] * numOfPolygon  # Equivalent to CArray<uint>

        incre = 0
        for j in range(numOfPolygon):
            counter2[incre] = 4
            incre += 1

        polygonIndices = []  # Equivalent to CArrayList<uint>
        incre = 0
        for nz in range(segmentZ + 1):
            for nx in range(segmentX + 1):
                x = nx * dx - lengthX / 2
                z = nz * dz - lengthZ / 2
                vertices.append(RV([x, 0.0, z]))

        for nz in range(segmentZ):
            for nx in range(segmentX):
                v0 = nx + nz * (segmentX + 1)
                v1 = nx + 1 + nz * (segmentX + 1)
                v2 = nx + 1 + (nz + 1) * (segmentX + 1)
                v3 = nx + (nz + 1) * (segmentX + 1)

                if (nx + nz) % 2 == 0:
                    polygonIndices[incre] = [v3, v2, v1, v0]
                else:
                    polygonIndices[incre] = [v2, v1, v0, v3]

                incre += 1

        polySet = self.statePolygonSet.getDataPtr()

        polySet.setPoints(vertices)
        polySet.setPolygons(polygonIndices)
        polySet.update()

        polygonIndices.clear()

        ts = self.stateTriangleSet.getData()
        polySet.turnIntoTriangleSet(ts)

        qs = self.stateQuadSet.getData()
        polySet.turnIntoQuadSet(qs)

        vertices.clear()


    def boundingBox(self):
        center = self.varLocation().getValue()
        rot = self.varRotation().getValue()
        scale = self.varScale().getValue()
        q = self.computeQuaternion()

print(isinstance(PlaneModel,dyno.Node ))
scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([2,2,2]))
scn.setLowerBound(dyno.Vector3f([-2,-2,-2]))


plane = PlaneModel()
scn.addNode(plane)

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()

