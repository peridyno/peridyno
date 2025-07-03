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
    UbiGUI
    WtGUI
)
add_example(Wt_Preset WtGUI LIB_DEPENDENCY)
