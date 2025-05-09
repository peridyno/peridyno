import PyPeridyno as dyno

scn = dyno.SceneGraph()

ocean = dyno.Ocean3f()
scn.add_node(ocean)

patch = dyno.OceanPatch3f()
scn.add_node(patch)
patch.var_wind_type().set_value(5)
patch.var_patch_size().set_value(128)
patch.connect(ocean.import_ocean_patch())

wake = dyno.Wake3f()
scn.add_node(wake)
wake.var_water_level().set_value(4)
wake.var_length().set_value(128)
wake.var_magnitude().set_value(0.2)
wake.connect(ocean.import_capillary_waves())

mapper = dyno.HeightFieldToTriangleSet3f()

ocean.state_height_field().connect(mapper.in_height_field())
ocean.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0,0.2,1))
sRender.var_use_vertex_normal().set_value(False)
sRender.var_alpha().set_value(0.6)
mapper.out_triangle_set().connect(sRender.in_triangle_set())
ocean.graphics_pipeline().push_module(sRender)

boat = dyno.Vessel3f()
scn.add_node(boat)
boat.var_density().set_value(150)
boat.var_barycenter_offset().set_value(dyno.Vector3f([0,0,-0.5]))
boat.state_velocity().set_value(dyno.Vector3f([0,0,0]))

steer = dyno.Steer3f()
boat.state_velocity().connect(steer.in_velocity())
boat.state_angular_velocity().connect(steer.in_angular_velocity())
boat.state_quaternion().connect(steer.in_quaternion())
boat.animation_pipeline().push_module(steer)

coupling = dyno.RigidWaterCoupling3f()
scn.add_node(coupling)
boat.connect(wake.import_vessel())
boat.connect(coupling.import_vessels())
ocean.connect(coupling.import_ocean())

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()
