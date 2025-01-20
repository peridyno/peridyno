import PyPeridyno as dyno

scn = dyno.SceneGraph()
scn.set_upper_bound(dyno.Vector3f([15.5, 15, 15.5]))
scn.set_lower_bound(dyno.Vector3f([-15.5, -15, -15.5]))

obj = dyno.ObjLoader3f()
scn.add_node(obj)

obj.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "Building/YXH_Poly.obj"))
obj.var_scale().set_value(dyno.Vector3f([0.2,0.2,0.2]))

extrude = dyno.PolyExtrude3f()
scn.add_node(extrude)
obj.out_triangle_set().connect(extrude.in_triangle_set())
extrude.var_primitive_id().set_value(" 0-109 292-413 430-836 1461-1486 1558-1647 1658-1709 1762-1842 1909-2132 2134-3016 3151-3154 3253-3326 3496 4816 4819 4828 5039 5956 7382-7383 7389-7392 7408 7413-7416 7722-7863 7871-7925 7935-8099 8102-8103 8140-8225 8245-8249 ")
extrude.var_distance().set_value(0.15)

extrude2 = dyno.PolyExtrude3f()
scn.add_node(extrude2)
extrude.state_triangle_set().promote_output().connect(extrude2.in_triangle_set())
extrude2.var_primitive_id().set_value(" 837-1460 1487-1557 1648-1657 6422-6423 6438-6441 6483 6487-6488 6496 6519-6524 6595 6598-6606 6944 7654-7655 8126-8127")
extrude2.var_distance().set_value(0.3)

extrude3 = dyno.PolyExtrude3f()
scn.add_node(extrude3)
extrude2.state_triangle_set().promote_output().connect(extrude3.in_triangle_set())
extrude3.var_primitive_id().set_value(" 110-290 ")
extrude3.var_distance().set_value(0.4)

pt = dyno.ObjPoint3f()
scn.add_node(pt)
pt.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "Building/Tree_Scatter.obj"))

tree = dyno.ObjLoader3f()
scn.add_node(tree)
tree.var_file_name().set_value(dyno.FilePath(dyno.get_asset_path() + "Building/Tree_Poly.obj"))

copy = dyno.CopyToPoint3f()
scn.add_node(copy)
tree.out_triangle_set().connect(copy.in_triangle_set_in())
pt.out_point_set().promote_output().connect(copy.in_triangle_set_in())


group = dyno.Group3f()
scn.add_node(group)
group.var_primitive_id().set_value(" 1 2-8 19-25")
group.var_edge_id().set_value(" 3-8 12 16 25-27")
group.var_point_id().set_value(" 10 15-20 30 35 38-40")

app = dyno.GlfwApp()
app.set_scenegraph(scn)
app.initialize(1366, 768, True)
app.main_loop()
