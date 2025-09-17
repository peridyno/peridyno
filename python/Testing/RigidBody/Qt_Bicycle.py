import QtPathHelper
import PyPeridyno as dyno

class PythonSteer(dyno.KeyboardInputModule):
    def __init__(self):
        super().__init__()
        self.var_Strength = dyno.FVarReal(1.0, "Strength", "Strength", dyno.FieldTypeEnum.Param, self)
        self.in_Velocity = dyno.FVar3f("Velocity", "Velocity", dyno.FieldTypeEnum.In, self)
        self.in_AngularVelocity = dyno.FVar3f("AngularVelocity", "Angular velocity", dyno.FieldTypeEnum.In, self)
        self.in_Quaternion = dyno.FVarQuatReal("Quaternion", "Rotation", dyno.FieldTypeEnum.In, self)

    @property
    def varStrength(self):
        return self.var_Strength

    @property
    def inVelocity(self):
        return self.in_Velocity

    @property
    def inAngularVelocity(self):
        return self.in_AngularVelocity

    @property
    def inQuaternion(self):
        return self.in_Quaternion

    def onEvent(self, event):
        quat = self.inQuaternion.getData()
        vel = self.inVelocity.getData()
        omega = self.inAngularVelocity.getData()

        rot = quat.toMatrix3x3()

        vel_prime = rot.transpose() * vel
        omega_prime = rot.transpose() * omega

        strength = self.varStrength.getValue()

        if event.key == dyno.PKeyboardType.PKEY_A:
            omega_prime[1] += strength
        elif event.key == dyno.PKeyboardType.PKEY_S:
            vel_prime[2] *= 0.95
        elif event.key == dyno.PKeyboardType.PKEY_D:
            omega_prime[1] -= strength
        elif event.key == dyno.PKeyboardType.PKEY_W:
            vel_prime[2] += strength
            vel_prime[2] = 5 if vel_prime[2] > 5 else vel_prime[2]
            vel_prime[2] = -5 if vel_prime[2] < -5 else vel_prime[2]

        self.inVelocity.setValue(rot * vel_prime)
        self.inAngularVelocity.setValue(rot * omega_prime)

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

