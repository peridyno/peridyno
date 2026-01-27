if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        RigidBody
        ParticleSystem
        Peridynamics
        SemiAnalyticalScheme
        Multiphysics
        UbiGUI
        ObjIO)
    add_example(Qt_JeepLandscape Modeling LIB_DEPENDENCY)
endif()