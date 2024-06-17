import PyPeridyno as dyno

scene = dyno.SceneGraph()
scene.set_lower_bound(dyno.Vector3f([-1.5, 0, -1.5]))
scene.set_upper_bound(dyno.Vector3f([1.5, 3, 1.5]))

staticTriangularMesh = dyno.StaticTriangularMesh3f()
staticTriangularMesh.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "cloth_shell/ball/ball_model.obj"))

boundary = dyno.VolumeBoundary3f()
boundary.load_cube(dyno.Vector3f([-1.5, 0, -1.5]), dyno.Vector3f([1.5, 3, 1.5]), 0.005, True)
boundary.load_sdf(dyno.get_asset_path() + "cloth_shell/ball/ball_small_size_15.sdf", False)

cloth = dyno.CodimensionalPD3f(0.15, 500, 0.0005, 0.001, "default")
cloth.set_dt(0.001)
cloth.load_surface(dyno.get_asset_path() + "cloth_shell/cloth_size_17_alt/cloth_40k_3.obj")
cloth.connect(boundary.import_triangular_systems())
cloth.set_grad_ite_eps(0)
cloth.set_max_ite_number(10)
surfaceRendererCloth = dyno.GLSurfaceVisualModule()
surfaceRendererCloth.set_color(dyno.Color(0.4, 0.4, 1.0))

surfaceRenderer = dyno.GLSurfaceVisualModule()
surfaceRenderer.set_color(dyno.Color(0.4, 0.4, 0.4))
surfaceRenderer.var_use_vertex_normal().set_value(True)
cloth.state_triangle_set().connect(surfaceRendererCloth.in_triangle_set())
staticTriangularMesh.state_triangle_set().connect(surfaceRenderer.in_triangle_set())
cloth.graphics_pipeline().push_module(surfaceRendererCloth)
staticTriangularMesh.graphics_pipeline().push_module(surfaceRenderer)
cloth.set_visible(True)
staticTriangularMesh.set_visible(True)
scene.print_node_info(True)
scene.print_simulation_info(True)

scene.add_node(staticTriangularMesh)
scene.add_node(boundary)
scene.add_node(cloth)
