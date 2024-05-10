import os

import PyPeridyno as dyno


def filePath(str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = "../../../data/" + str
    file_path = os.path.join(script_dir, relative_path)
    if os.path.isfile(file_path):
        print(file_path)
        return file_path
    else:
        print(f"File not found: {file_path}")
        return -1


scn = dyno.SceneGraph()

ocean = dyno.Ocean3f()
patch = dyno.OceanPatch3f()
patch.var_wind_type().set_value(5)
patch.var_patch_size().set_value(128.0)
patch.connect(ocean.import_ocean_patch())

wake = dyno.Wake3f()
wake.var_water_level().set_value(4)
wake.var_length().set_value(128.0)
wake.var_magnitude().set_value(0.2)
wake.connect(ocean.import_capillary_waves())

mapper = dyno.HeightFieldToTriangleSet3f()

ocean.state_height_field().connect(mapper.in_height_field())
ocean.graphics_pipeline().push_module(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.set_color(dyno.Color(0, 0.2, 1))
sRender.var_use_vertex_normal().set_value(True)
sRender.var_alpha().set_value(0.6)
mapper.out_triangle_set().connect(sRender.in_triangle_set())
ocean.graphics_pipeline().push_module(sRender)

gltf = dyno.GltfLoader3f()
gltf.var_file_name().set_value(dyno.FilePath(filePath("gltf/SailBoat/SailBoat.gltf")))

boat = dyno.Vessel3f()
boat.var_density().set_value(150)
boat.var_barycenter_offset().set_value(dyno.Vector3f([0, 0, -0.5]))
boat.state_velocity().set_value(dyno.Vector3f([0, 0, 0]))
boat.var_envelope_name().set_value(dyno.FilePath(filePath("gltf/SailBoat/SailBoat_boundary.obj")))

gltf.state_texture_mesh().connect(boat.in_texture_mesh())
gltf.set_visible(False)

steer = dyno.Steer3f()
boat.state_velocity().connect(steer.in_velocity())
boat.state_angular_velocity().connect(steer.in_angular_velocity())
boat.state_quaternion().connect(steer.in_quaternion())
boat.animation_pipeline().push_module(steer)

coupling = dyno.Coupling3f()
boat.connect(wake.import_vessel())
boat.connect(coupling.import_vessel())
ocean.connect(coupling.import_ocean())

scn.add_node(ocean)
scn.add_node(patch)
scn.add_node(wake)
scn.add_node(gltf)
scn.add_node(boat)
scn.add_node(coupling)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.render_window().get_camera().set_unit_scale(10)
app.main_loop()
