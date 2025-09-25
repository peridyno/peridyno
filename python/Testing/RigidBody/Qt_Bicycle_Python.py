import math

import QtPathHelper
import PyPeridyno as dyno

class TypeInfo:
    @staticmethod
    def cast(TA, b):
        """模拟 shared_ptr 的 dynamic_pointer_cast"""
        if b is None:
            return None

        # 检查类型兼容性
        if hasattr(b, '__class__') and isinstance(b, TA):
            return b

        # 如果 b 有 as_xxx 或 cast 方法，尝试使用
        cast_method = getattr(b, 'cast', None)
        if cast_method and callable(cast_method):
            try:
                return cast_method(TA)
            except:
                pass

        return None

    @staticmethod
    def cast_shared(TA, b):
        """专门处理 shared_ptr 的转换"""
        return TypeInfo.cast(TA, b)


class PyKeyDriver(dyno.KeyboardInputModule):
    def __init__(self):
        super().__init__()
        self.var_HingeKeyConfig = dyno.FVarKey2HingeConfig(dyno.Key2HingeConfig(), "HingeKeyConfig", "Config", dyno.FieldTypeEnum.Param, self)
        self.in_Reset = dyno.FVarBool("Reset", "Reset", dyno.FieldTypeEnum.In, self)
        self.in_Topology = dyno.FInstanceDiscreteElements3f("Topology", "Topology", dyno.FieldTypeEnum.In, self)
        self.hingeAngle = {}
        self.speed = 0.0
        self.varCacheEvent().setValue(False)

    @property
    def varHingeKeyConfig(self):
        return self.var_HingeKeyConfig

    @property
    def inReset(self):
        return self.in_Reset

    @property
    def inTopology(self):
        return self.in_Topology

    def onEvent(self, event):
        if self.inReset.getValue():
            self.hingeAngle.clear()
            self.inReset.setValue(False)
            self.speed = 0.0

        topo = self.inTopology.getDataPtr()
        d_hinge = topo.hingeJoints()
        c_hinge = dyno.CArrayHingeJoint()
        c_hinge.assign(d_hinge)

        keyConfig = self.varHingeKeyConfig.getValue()

        stepAngle = math.pi / 50
        currentHingeActions = []
        key2HingeActionIterator = keyConfig.key2Hinge.get(event.key)

        if key2HingeActionIterator is not None:
            currentHingeActions = key2HingeActionIterator

        if len(currentHingeActions) > 0:
            if event.key == dyno.PKeyboardType.PKEY_A or event.key == dyno.PKeyboardType.PKEY_D:
                for action in currentHingeActions:
                    keyJointID = action.joint
                    keyValue = action.value

                    if keyJointID in self.hingeAngle:
                        self.hingeAngle[keyJointID] = self.hingeAngle[keyJointID] + keyValue * stepAngle
                    else:
                        self.hingeAngle[keyJointID] = keyValue * stepAngle

                    min_angle = self.hingeAngle[keyJointID]
                    max_angle = max(min(self.hingeAngle[keyJointID] + math.pi / 360.0, 2 * math.pi), -2 * math.pi)
                    c_hinge[keyJointID].setRange(min_angle, max_angle)

            elif event.key == dyno.PKeyboardType.PKEY_W or event.key == dyno.PKeyboardType.PKEY_S:
                if event.key == dyno.PKeyboardType.PKEY_W:
                    self.speed += 0.5
                elif event.key == dyno.PKeyboardType.PKEY_S:
                    self.speed -= 0.5

                self.speed = max(min(self.speed, 5.0), 0.0)

                for action in currentHingeActions:
                    keyJointID = action.joint
                    keyValue = action.value
                    print(keyJointID)
                    print(c_hinge.size())
                    c_hinge[keyJointID].setMoter(self.speed * keyValue)

        d_hinge.assign(c_hinge)



scn = dyno.SceneGraph()

bike = dyno.Bicycle3f()
scn.addNode(bike)

multisystem = dyno.MultibodySystem3f()
scn.addNode(multisystem)
driver = PyKeyDriver()
# driver = dyno.KeyDriver3f()
multisystem.stateTopology().connect(driver.inTopology)
multisystem.animationPipeline().pushModule(driver)
bike.outReset().connect(driver.inReset)

keyConfig = dyno.Key2HingeConfig()
keyConfig.addMap(dyno.PKeyboardType.PKEY_W, 1, 1)
keyConfig.addMap(dyno.PKeyboardType.PKEY_S, 1, -1)

keyConfig.addMap(dyno.PKeyboardType.PKEY_D, 2, 1)
keyConfig.addMap(dyno.PKeyboardType.PKEY_A, 2, -1)
driver.varHingeKeyConfig.setValue(keyConfig)

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

