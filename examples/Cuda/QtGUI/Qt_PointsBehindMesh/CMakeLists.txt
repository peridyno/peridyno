if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        RigidBody
        ParticleSystem
        SemiAnalyticalScheme
        Modeling
        QtGUI
        ObjIO)
    add_example(Qt_PointsBehindMesh Qt LIB_DEPENDENCY)
endif()