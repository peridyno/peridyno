import PyPeridyno as dyno

class PythonSteer(dyno.KeyboardInputModule):
    def __init__(self):
        super().__init__()
        self.var_Strength = dyno.FVarf(1.0, "Strength", "Strength", dyno.FieldTypeEnum.Param, self)
        self.in_Velocity = dyno.FVarf("Velocity", "Velocity", dyno.FieldTypeEnum.In, self)
        self.in_AngularVelocity = dyno.FVarf("AngularVelocity", "Angular velocity", dyno.FieldTypeEnum.In, self)
        self.in_Quaternion = dyno.FVarf("Quaternion", "Rotation", dyno.FieldTypeEnum.In, self)

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
        print("python onEvent")

        # quat = self.inQuaternion.getData()
        # vel = self.inVelocity.getData()
        # omega = self.inAngularVelocity.getData()
        # rot = quat.toMatrix3x3()
        #
        # vel_prime = rot.transpose() * vel
        # omega_prime = rot.transpose() * omega
        #
        # strength = self.varStrength.getValue()
        #
        # if event.key == dyno.PKeyboardType.PKEY_A:
        #     print("python botton A")
        #     omega_prime.y += strength
        # elif event.key == dyno.PKeyboardType.PKEY_S:
        #     vel_prime[2] *= 0.95
        # elif event.key == dyno.PKeyboardType.PKEY_D:
        #     omega_prime.y -= strength
        # elif event.key == dyno.PKeyboardType.PKEY_W:
        #     vel_prime[2] += strength
        #     vel_prime[2] = 5 if vel_prime[2] > 5 else vel_prime[2]
        #     vel_prime[2] = -5 if vel_prime[2] < -5 else vel_prime[2]
        #
        # self.inVelocity.setValue(rot * vel_prime)
        # self.inAngularVelocity.setValue(rot * omega_prime)

scn = dyno.SceneGraph()

ocean = dyno.Ocean3f()
ocean.varExtentX().setValue(2)
ocean.varExtentZ().setValue(2)
scn.addNode(ocean)

patch = dyno.OceanPatch3f()
scn.addNode(patch)
patch.varWindType().setValue(5)
patch.varPatchSize().setValue(128)
patch.connect(ocean.importOceanPatch())
patch.varResolution().setValue(512)

wake = dyno.Wake3f()
scn.addNode(wake)
wake.varWaterLevel().setValue(4)
wake.varLength().setValue(128)
wake.varMagnitude().setValue(0.2)
wake.connect(ocean.importCapillaryWaves())

mapper = dyno.HeightFieldToTriangleSet3f()

ocean.stateHeightField().connect(mapper.inHeightField())
ocean.graphicsPipeline().pushModule(mapper)

sRender = dyno.GLSurfaceVisualModule()
sRender.setColor(dyno.Color(0,0.2,1))
sRender.varUseVertexNormal().setValue(False)
sRender.varAlpha().setValue(0.6)
mapper.outTriangleSet().connect(sRender.inTriangleSet())
ocean.graphicsPipeline().pushModule(sRender)

boat = dyno.Vessel3f()
scn.addNode(boat)
boat.varDensity().setValue(150)
boat.varBarycenterOffset().setValue(dyno.Vector3f([0,0,-0.5]))
boat.stateVelocity().setValue(dyno.Vector3f([0,0,0]))
boat.varEnvelopeName().setValue(dyno.FilePath(dyno.getAssetPath() + "obj/boat_boundary.obj"))
boat.varTextureMeshName().setValue(dyno.FilePath(dyno.getAssetPath() + "gltf/SailBoat/SailBoat.gltf"))

# steer = dyno.Steer3f()
# boat.stateVelocity().connect(steer.inVelocity())
# boat.stateAngularVelocity().connect(steer.inAngularVelocity())
# boat.stateQuaternion().connect(steer.inQuaternion())
# boat.animationPipeline().pushModule(steer)

pySteer = PythonSteer()
boat.stateVelocity().connect(pySteer.inVelocity)
boat.stateAngularVelocity().connect(pySteer.inAngularVelocity)
boat.stateQuaternion().connect(pySteer.inQuaternion)
boat.animationPipeline().pushModule(pySteer)

coupling = dyno.RigidWaterCoupling3f()
scn.addNode(coupling)
boat.connect(wake.importVessel())
boat.connect(coupling.importVessels())
ocean.connect(coupling.importOcean())


app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.setWindowTitle("Empty GUI")
app.mainLoop()

