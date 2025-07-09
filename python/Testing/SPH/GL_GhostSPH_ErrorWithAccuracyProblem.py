import PyPeridyno as dyno
import numpy as np

def floatRange(start, stop, step):
    x = start
    while x <= stop:
        yield x
        x += step

def createFluidParticles():

    fluid = dyno.ParticleFluid3f()

    hostPos = []
    hostVel = []

    lowx = -0.1
    lowy = 0
    lowz = -0.1

    highx = 0.1
    highy = 0.1
    highz = 0.1

    s = 0.005
    m_iExt = 0

    omega = 1.0
    halfS = -s / 2.0

    num = 0

    x = lowx
    y = lowy
    z = lowz

    for x in floatRange(lowx, highx, s):
        for y in floatRange(lowy, highy, s):
            for z in floatRange(lowz, highz, s):
                p = dyno.Vector3f([x,y,z])
                hostPos.append(p)
                hostVel.append(dyno.Vector3f([0,0,0]))

    fluid.statePosition().assign(hostPos)
    fluid.stateVelocity().assign(hostVel)

    hostPos.clear()
    hostVel.clear()
    return fluid

def createGhostParticles():
    ghost = dyno.GhostParticles3f()

    hostPos = dyno.VectorVec3f()
    hostVel = dyno.VectorVec3f()
    hostForce = dyno.VectorVec3f()
    hostNormal = dyno.VectorVec3f()
    hostAttribute = dyno.VectorAttribute()

    low = dyno.Vector3f([-0.2, -0.015, -0.2])
    high = dyno.Vector3f([0.2, -0.005, 0.2])

    lowx = -0.2
    lowy = -0.015
    lowz = -0.2

    highx = 0.2
    highy = -0.005
    highz = 0.2

    s = 0.005
    m_iExt = 0

    omega = 1.0
    halfS = -s / 2.0

    num = 0
    for x in np.arange(lowx - m_iExt * s, highx + m_iExt * s + s, s):
        for y in np.arange(lowy - m_iExt * s, highy + m_iExt * s, s):
            for z in np.arange(lowz - m_iExt * s, highz + m_iExt * s, s):
                print(x, y, z)
                attri = dyno.Attribute()
                attri.setFluid()
                attri.setDynamic()

                hostPos.append(dyno.Vector3f([x, y, z]))
                hostVel.append(dyno.Vector3f([0, 0, 0]))
                hostForce.append(dyno.Vector3f([0, 0, 0]))
                hostNormal.append(dyno.Vector3f([0, 1, 0]))
                hostAttribute.append(attri)

    ghost.statePosition().resize(num)
    ghost.stateVelocity().resize(num)

    ghost.stateNormal().resize(num)
    ghost.stateAttribute().resize(num)

    ghost.statePosition().assign(hostPos)
    ghost.stateVelocity().assign(hostVel)
    ghost.stateForce().assign(hostForce)
    ghost.stateNormal().assign(hostNormal)
    ghost.stateAttribute().assgin(hostAttribute)

    hostPos.clear()
    hostVel.clear()
    hostForce.clear()
    hostNormal.clear()
    hostAttribute.clear()
    return ghost


scn = dyno.SceneGraph()
scn.setUpperBound(dyno.Vector3f([0.5,1,0.5]))
scn.setLowerBound(dyno.Vector3f([-0.5,0,-0.5]))

cubeBoundary = dyno.CubeModel3f()
scn.addNode(cubeBoundary)
cubeBoundary.varLocation().setValue(dyno.Vector3f([0,0.5,0]))
cubeBoundary.varLength().setValue(dyno.Vector3f([1,1,1]))

cube2vol = dyno.BasicShapeToVolume3f()
scn.addNode(cube2vol)
cube2vol.varGridSpacing().setValue(0.02)
cube2vol.varInerted().setValue(True)
cubeBoundary.connect(cube2vol.importShape())

boundary = dyno.VolumeBoundary3f()
scn.addNode(boundary)
cube2vol.connect(boundary.importVolumes())

fluid = createFluidParticles()
scn.addNode(fluid)
ghost = createGhostParticles()
scn.addNode(ghost)

incompressibleFluid = dyno.GhostFluid3f()
scn.addNode(incompressibleFluid)
incompressibleFluid.setDt(0.001)
fluid.connect(incompressibleFluid.importInitialStates())
ghost.connect(incompressibleFluid.importBoundaryParticles())

incompressibleFluid.connect(boundary.importParticleSystems())

app = dyno.GlfwApp()
app.setSceneGraph(scn)
app.initialize(1920, 1080, True)
app.mainLoop()
