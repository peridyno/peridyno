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
    add_example(Qt_GUI_Plugin Qt LIB_DEPENDENCY)
endif()