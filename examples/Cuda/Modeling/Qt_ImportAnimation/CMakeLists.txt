if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        RigidBody
        ParticleSystem
        Peridynamics
        SemiAnalyticalScheme
        Multiphysics
        SkeletonLoader
        QtGUI)
    add_example(Qt_ImportAnimation Modeling LIB_DEPENDENCY)
endif()