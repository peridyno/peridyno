import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([15.5, 15, 15.5]))
scn.setLowerBound(dyno.Vector3f([-15.5, -15, -15.5]))

obj = dyno.ObjLoader3f()
scn.addNode(obj)

obj.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Building/YXH_Poly.obj"))
obj.varScale().setValue(dyno.Vector3f([0.2,0.2,0.2]))

extrude = dyno.PolyExtrude3f()
scn.addNode(extrude)
obj.outTriangleSet().connect(extrude.inTriangleSet())
extrude.varPrimitiveId().setValue(" 0-109 292-413 430-836 1461-1486 1558-1647 1658-1709 1762-1842 1909-2132 2134-3016 3151-3154 3253-3326 3496 4816 4819 4828 5039 5956 7382-7383 7389-7392 7408 7413-7416 7722-7863 7871-7925 7935-8099 8102-8103 8140-8225 8245-8249 ")
extrude.varDistance().setValue(0.15)

extrude2 = dyno.PolyExtrude3f()
scn.addNode(extrude2)
extrude.stateTriangleSet().promoteOuput().connect(extrude2.inTriangleSet())
extrude2.varPrimitiveId().setValue(" 837-1460 1487-1557 1648-1657 6422-6423 6438-6441 6483 6487-6488 6496 6519-6524 6595 6598-6606 6944 7654-7655 8126-8127")
extrude2.varDistance().setValue(0.3)

extrude3 = dyno.PolyExtrude3f()
scn.addNode(extrude3)
extrude2.stateTriangleSet().promoteOuput().connect(extrude3.inTriangleSet())
extrude3.varPrimitiveId().setValue(" 110-290 ")
extrude3.varDistance().setValue(0.4)

pt = dyno.ObjPoint3f()
scn.addNode(pt)
pt.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Building/Tree_Scatter.obj"))

tree = dyno.ObjLoader3f()
scn.addNode(tree)
tree.varFileName().setValue(dyno.FilePath(dyno.getAssetPath() + "Building/Tree_Poly.obj"))

copy = dyno.CopyToPoint3f()
scn.addNode(copy)
tree.outTriangleSet().connect(copy.inTriangleSetIn())
pt.outPointSet().promoteOuput().connect(copy.inTriangleSetIn())


group = dyno.Group3f()
scn.addNode(group)
group.varPrimitiveId().setValue(" 1 2-8 19-25")
group.varEdgeId().setValue(" 3-8 12 16 25-27")
group.varPointId().setValue(" 10 15-20 30 35 38-40")

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1366, 768, True)
app.mainLoop()
