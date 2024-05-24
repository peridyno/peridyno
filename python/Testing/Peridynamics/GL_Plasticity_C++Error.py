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

elastoplasticBody = dyno.ElastoplasticBody3f()
elastoplasticBody.set_visible(False)
elastoplasticBody.load_particles(dyno.Vector3f([-1.1, -1.1, -1.1]), dyno.Vector3f([1.15, 1.15, 1.15]), 0.1)
elastoplasticBody.scale(0.05)
elastoplasticBody.translate(dyno.Vector3f([0.3, 0.2, 0.5]))

surfaceMeshLoader = dyno.SurfaceMeshLoader3f()
surfaceMeshLoader.var_file_name().set_value(dyno.FilePath(filePath("standard/standard_cube20.obj")))
surfaceMeshLoader.var_scale().set_value(dyno.Vector3f([0.05, 0.05, 0.05]))
surfaceMeshLoader.var_location().set_value(dyno.Vector3f([0.3, 0.2, 0.5]))

topoMapper = dyno.PointSetToTriangleSet3f()

outTop = elastoplasticBody.state_point_set().promote_output()
outTop.connect(topoMapper.in_point_set())
surfaceMeshLoader.out_triangle_set().connect(topoMapper.in_initial_shape())

surfaceVisualizer = dyno.GLSurfaceVisualNode3f()
topoMapper.out_shape().connect(surfaceVisualizer.in_triangle_set())

elasticBody = dyno.ElasticBody3f()
elasticBody.set_visible(False)
elasticBody.load_particles(dyno.Vector3f([-1.1, -1.1, -1.1]), dyno.Vector3f([1.15, 1.15, 1.15]), 0.1)
elasticBody.scale(0.05)
elasticBody.translate(dyno.Vector3f([0.5, 0.2, 0.5]))

surfaceMeshLoader2 = dyno.SurfaceMeshLoader3f()
surfaceMeshLoader2.var_file_name().set_value(dyno.FilePath(filePath("standard/standard_cube20.obj")))
surfaceMeshLoader2.var_scale().set_value(dyno.Vector3f([0.05, 0.05, 0.05]))
surfaceMeshLoader2.var_location().set_value(dyno.Vector3f([0.3, 0.2, 0.5]))

topoMapper2 = dyno.PointSetToTriangleSet3f()

outTop2 = elastoplasticBody.state_point_set().promote_output()
outTop2.connect(topoMapper2.in_point_set())
surfaceMeshLoader2.out_triangle_set().connect(topoMapper2.in_initial_shape())

surfaceVisualizer2 = dyno.GLSurfaceVisualNode3f()
topoMapper2.out_shape().connect(surfaceVisualizer2.in_triangle_set())

boundary = dyno.StaticBoundary3f()
boundary.load_cube(dyno.Vector3f([0,0,0]), dyno.Vector3f([1,1,1]), 0.005, True)
elastoplasticBody.connect(boundary.import_particle_systems())
elasticBody.connect(boundary.import_particle_systems())


scn.add_node(elastoplasticBody)
scn.add_node(surfaceMeshLoader)
scn.add_node(topoMapper)
scn.add_node(surfaceVisualizer)
scn.add_node(elasticBody)
scn.add_node(surfaceMeshLoader2)
scn.add_node(boundary)
scn.add_node(topoMapper2)
scn.add_node(surfaceVisualizer2)

app = dyno.GLfwApp()
app.set_scenegraph(scn)
app.initialize(1920, 1080, True)
app.main_loop()

