if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Core 
        Framework 
        Modeling
        RigidBody
        ParticleSystem
        Peridynamics
        SemiAnalyticalScheme
        Multiphysics
        QtGUI
        Volume
        ObjIO)
    add_example(Qt_PointSampling Modeling LIB_DEPENDENCY)
endif()