import PyPeridyno as dyno

# createScene
scn = dyno.SceneGraph()

# 为什么要在主函数里面定义Point\Distance，那我怎么去
point

scn.add_node(point)
scn.add_node(cube)
scn.add_node(calculation)
point.connect(calculation)
cube.connect(calculation)

mapper = dyno.DiscreteElementsToTriangleSet3f()
rigid.current_topology().connect(mapper.in_discreteElements())
rigid.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule3f()
sRender.set_color(dyno.Color(1, 1, 0))
mapper.out_triangleSet().connect(sRender.in_triangleSet())
rigid.graphics_pipeline().push_module(sRender)

app = dyno.GLApp()
app.set_scenegraph(scn)
app.initialize(1280, 768, True)
app.main_loop()
