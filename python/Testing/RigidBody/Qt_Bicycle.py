import QtPathHelper
import PyPeridyno as dyno

scn = dyno.SceneGraph()

bike = dyno.Bicycle3f()
scn.addNode(bike)

multisystem = dyno.MultibodySystem3f()
scn.addNode(multisystem)
driver = dyno.KeyDriver3f()
multisystem.stateTopology().connect(driver.inTopology())
multisystem.animationPipeline().pushModule(driver)
bike.outReset().connect(driver.inReset())

keyConfig = dyno.Key2HingeConfig()
keyConfig.addMap(dyno.PKeyboardType.PKEY_W, 1, 1)
keyConfig.addMap(dyno.PKeyboardType.PKEY_S, 1, -1)

keyConfig.addMap(dyno.PKeyboardType.PKEY_D, 2, 1)
keyConfig.addMap(dyno.PKeyboardType.PKEY_A, 2, -1)
driver.varHingeKeyConfig().setValue(keyConfig)

plane = dyno.PlaneModel3f()
scn.addNode(plane)
bike.connect(multisystem.importVehicles())
plane.stateTriangleSet().connect(multisystem.inTriangleSet())
plane.varLengthX().setValue(120)
plane.varLengthZ().setValue(120)
plane.varLocation().setValue(dyno.Vector3f([0,-0.5,0]))


#app = dyno.QtApp()
app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
#app.renderWindow().getCamera().setUnitScale(3)
app.mainLoop()

