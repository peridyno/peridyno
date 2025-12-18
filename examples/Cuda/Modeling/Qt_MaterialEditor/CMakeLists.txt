if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        RigidBody
        HeightField
        ParticleSystem
        Peridynamics
        DualParticleSystem
        SemiAnalyticalScheme
        Multiphysics
        UbiGUI)
    add_example(Qt_MaterialEditor Modeling LIB_DEPENDENCY)
endif()