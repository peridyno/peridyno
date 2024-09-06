if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        RigidBody
        ParticleSystem
        Peridynamics
        DualParticleSystem
        SemiAnalyticalScheme
        Multiphysics
        QtGUI)
    add_example(Qt_GUI_Plugin Qt LIB_DEPENDENCY)
endif()