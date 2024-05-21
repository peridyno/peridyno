import os

import PyPeridyno as dyno


def filePath(str):
    script_dir = os.getcwd()
    relative_path = "../../../../data/" + str
    file_path = os.path.join(script_dir, relative_path)
    if os.path.isfile(file_path):
        print(file_path)
        return file_path
    else:
        print(f"File not found: {file_path}")
        return -1


scene = dyno.SceneGraph()
scene.set_gravity(dyno.Vector3f([0, -9.8, 0]))

emitter = dyno.CircularEmitter3f()
emitter.var_location().set_value(dyno.Vector3f([0,1,0]))

fluid = dyno.ParticleFluid3f()
emitter.connect(fluid.import_particle_emitters())

ptRender = dyno.GLPointVisualModule()
ptRender.var_point_size().set_value(0.002)
ptRender.set_color(dyno.Color(1, 0, 0))
ptRender.set_color_map_mode(ptRender.ColorMapMode.PER_VERTEX_SHADER)

calculateNorm = dyno.CalculateNorm3f()
colorMapper = dyno.ColorMapping3f()
colorMapper.var_max().set_value(5)
fluid.state_velocity().connect(calculateNorm.in_vec())
calculateNorm.out_norm().connect(colorMapper.in_scalar())

colorMapper.out_color().connect(ptRender.in_color())
fluid.state_point_set().connect(ptRender.in_point_set())

fluid.graphics_pipeline().push_module(calculateNorm)
fluid.graphics_pipeline().push_module(colorMapper)
fluid.graphics_pipeline().push_module(ptRender)

fluid.animation_pipeline().disable()

plane=dyno.PlaneModel3f()
plane.var_scale().set_value(dyno.Vector3f([2,0,2]))

sphere =dyno.SphereModel3f()
sphere.var_location().set_value(dyno.Vector3f([0,0.5,0]))
sphere.var_scale().set_value(dyno.Vector3f([0.2,0.2,0.2]))

merge = dyno.MergeTriangleSet3f()
plane.state_triangle_set().connect(merge.in_first())
sphere.state_triangle_set().connect(merge.in_second())

sfi = dyno.SemiAnalyticalSFINode3f()

fluid.connect(sfi.import_particle_systems())
merge.state_triangle_set().connect(sfi.in_triangle_set())


scene.add_node(emitter)
scene.add_node(fluid)
scene.add_node(plane)
scene.add_node(sphere)
scene.add_node(merge)
scene.add_node(sfi)