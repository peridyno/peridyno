import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_total_time(3.0)
scn.set_gravity(dyno.Vector3f([0, -9.8, 0]))
scn.set_upper_bound(dyno.Vector3f([0.5, 1, 4]))
scn.set_lower_bound(dyno.Vector3f([-0.5, 0, 4]))

velocity = dyno.Vector3f([0, 0, 6])
color = dyno.Color(1, 1, 1)

LocationBody = dyno.Vector3f([0, 0.01, -1])

anglurVel = dyno.Vector3f([100, 0, 0])
scale = dyno.Vector3f([0.4, 0.4, 0.4])

ObjJeep = dyno.ObjLoader3f()
scn.add_node(ObjJeep)
ObjJeep.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "Jeep/jeep_low.obj"))
ObjJeep.var_scale().set_value(scale)
ObjJeep.var_location().set_value(LocationBody)
ObjJeep.var_velocity().set_value(velocity)
glJeep = ObjJeep.graphics_pipeline().find_first_module_surface()
glJeep.set_color(color)

wheelPath = ["Jeep/Wheel_R.obj","Jeep/Wheel_R.obj","Jeep/Wheel_L.obj","Jeep/Wheel_R.obj"]
wheelSet = []

wheelLocation = []
wheelLocation.append(dyno.Vector3f([0.17, 0.1, 0.36]) + LocationBody)
wheelLocation.append(dyno.Vector3f([0.17, 0.1, -0.3]) + LocationBody)
wheelLocation.append(dyno.Vector3f([-0.17, 0.1, 0.36]) + LocationBody)
wheelLocation.append(dyno.Vector3f([-0.17, 0.1, -0.3]) + LocationBody)

for i in range(4):
    ObjWheel = dyno.ObjLoader3f()
    scn.add_node(ObjWheel)

    ObjWheel.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + wheelPath[i]))

    ObjWheel.var_scale().set_value(scale)
    ObjWheel.var_location().set_value(wheelLocation[i])
    ObjWheel.var_center().set_value(wheelLocation[i])

    ObjWheel.var_velocity().set_value(velocity)
    ObjWheel.var_angular_velocity().set_value(anglurVel)

    wheelSet.append(ObjWheel)

# Import Road
ObjRoad = dyno.ObjLoader3f()
scn.add_node(ObjRoad)
ObjRoad.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "Jeep/Road/Road.obj"))
ObjRoad.var_scale().set_value(dyno.Vector3f([0.04, 0.04, 0.04]))
ObjRoad.var_location().set_value(dyno.Vector3f([0, 0, 0.5]))
glRoad = ObjRoad.graphics_pipeline().find_first_module_surface()
glRoad.set_color(color)

# *************************************** Merge Model ***************************************//
# MergeWheel
mergeWheel = dyno.Merge3f()
scn.add_node(mergeWheel)
mergeWheel.var_update_mode().set_current_key(1)

wheelSet[0].out_triangle_set().connect(mergeWheel.in_triangle_set_01())
wheelSet[1].out_triangle_set().connect(mergeWheel.in_triangle_set_02())
wheelSet[2].out_triangle_set().connect(mergeWheel.in_triangle_set_03())
wheelSet[3].out_triangle_set().connect(mergeWheel.in_triangle_set_04())

# MergeRoad
mergeRoad = dyno.Merge3f()
scn.add_node(mergeRoad)
mergeRoad.var_update_mode().set_current_key(1)
mergeWheel.state_triangle_set().promote_output().connect(mergeRoad.in_triangle_set_01())
ObjRoad.out_triangle_set().connect(mergeRoad.in_triangle_set_03())

# Obj boundary
ObjBoundary = dyno.ObjLoader3f()
scn.add_node(ObjBoundary)
ObjBoundary.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "Jeep/Road/boundary.obj"))
ObjBoundary.var_scale().set_value(dyno.Vector3f([0.04, 0.04, 0.04]))
ObjBoundary.var_location().set_value(dyno.Vector3f([0, 0, 0.5]))
glBoundary = ObjBoundary.graphics_pipeline().find_first_module_surface()
glBoundary.set_color(color)

ObjBoundary.out_triangle_set().connect(mergeRoad.in_triangle_set_02())
ObjBoundary.graphics_pipeline().disable()
ObjJeep.out_triangle_set().connect(mergeRoad.in_triangle_set_04())

# SetVisible
mergeRoad.graphics_pipeline().disable()

# *************************************** Cube Sample ***************************************
# Cube
cube = dyno.CubeModel3f()
scn.add_node(cube)
cube.var_location().set_value(dyno.Vector3f([0, 0.025, 0.4]))
cube.var_length().set_value(dyno.Vector3f([0.35, 0.02, 3]))
cube.var_scale().set_value(dyno.Vector3f([2, 1, 1]))
cube.graphics_pipeline().disable()

cubeSampler = dyno.ShapeSampler3f()
scn.add_node(cubeSampler)
cubeSampler.var_sampling_distance().set_value(0.005)
cube.connect(cubeSampler.import_shape())
cubeSampler.graphics_pipeline().disable()

# MakeParticleSystem
particleSystem = dyno.MakeParticleSystem3f()
scn.add_node(particleSystem)

cubeSampler.state_point_set().promote_output().connect(particleSystem.in_points())

# *************************************** Fluid ***************************************//
# Particle fluid node
fluid = dyno.ParticleFluid3f()
scn.add_node(fluid)
particleSystem.connect(fluid.import_initial_states())

visualizer = dyno.GLPointVisualNode3f()
scn.add_node(visualizer)
ptrender = visualizer.graphics_pipeline().find_first_module_point()
ptrender.var_point_size().set_value(0.001)

fluid.state_point_set().promote_output().connect(visualizer.in_points())
fluid.state_velocity().promote_output().connect(visualizer.in_vector())

# SemiAnalyticalSFINode
meshBoundary = dyno.TriangularMeshBoundary3f()
scn.add_node(meshBoundary)
fluid.connect(meshBoundary.import_particle_systems())

mergeRoad.state_triangle_set().promote_output().connect(meshBoundary.in_triangle_set())

# Create a boundary
cubeBoundary = dyno.CubeModel3f()
scn.add_node(cubeBoundary)
cubeBoundary.var_location().set_value(dyno.Vector3f([0, 1, 0.75]))
cubeBoundary.var_length().set_value(dyno.Vector3f([2, 2, 4.5]))
cubeBoundary.set_visible(False)

cube2Vol = dyno.BasicShapeToVolume3f()
scn.add_node(cube2Vol)
cube2Vol.var_grid_spacing().set_value(0.02)
cube2Vol.var_inerted().set_value(True)
cubeBoundary.connect(cube2Vol.import_shape())

container = dyno.VolumeBoundary3f()
scn.add_node(container)
cube2Vol.connect(container.import_volumes())

fluid.connect(container.import_particle_systems())

# first Module
colormapping = visualizer.graphics_pipeline().find_first_module_color_mapping()
colormapping.var_max().set_value(1.5)

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1366, 768, True)
app.main_loop()
