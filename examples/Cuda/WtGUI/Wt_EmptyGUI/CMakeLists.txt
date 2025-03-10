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
    GlfwGUI
)
add_example(Wt_EmptyGUI WtGUI LIB_DEPENDENCY)
set_target_properties(Wt_EmptyGUI PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_FLAGS})
