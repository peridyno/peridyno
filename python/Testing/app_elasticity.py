import PyPeridyno as dyno

log = dyno.Log()
log.set_output("output.txt")
inst = dyno.SceneGraph.get_instance()
el = dyno.ParticleElasticBody3f()

el.load_particles("../../data/bunny/bunny_points.obj")
trans1 = dyno.Vector3f([0.5, 0.2, 0.5])
el.translate(trans1)

bound1 = dyno.StaticBoundary3f()
bound1.load_cube(dyno.Vector3f([0, 0, 0]), dyno.Vector3f([1, 1, 1]), 0.005, True, False)
inst.set_root_node(bound1)
bound1.add_particle_system(el)

r1 = dyno.PointRenderModule()
el.add_visual_module(r1)

app = dyno.GLApp()
app.create_window(800, 600)
app.main_loop()