if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        RigidBody
        ParticleSystem
        Peridynamics
        SemiAnalyticalScheme
        Multiphysics
        FBXLoader
        ObjIO
        QtGUI)
    add_example(Qt_ImportAnimation Modeling LIB_DEPENDENCY)
endif()