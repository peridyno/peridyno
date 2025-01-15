set(LIB_DEPENDENCY
    Modeling
    RigidBody
    HeightField
    ParticleSystem
    Peridynamics
    DualParticleSystem
    SemiAnalyticalScheme
    Multiphysics
    ObjIO
    WtGUI
)
add_example(Wt_EmptyGUI WtGUI LIB_DEPENDENCY)
