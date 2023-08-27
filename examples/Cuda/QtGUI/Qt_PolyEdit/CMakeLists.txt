if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        RigidBody
        ParticleSystem
        Peridynamics
        SemiAnalyticalScheme
        Multiphysics
        QtGUI
        ObjIO)
    add_example(Qt_PolyEdit Qt LIB_DEPENDENCY)
endif()