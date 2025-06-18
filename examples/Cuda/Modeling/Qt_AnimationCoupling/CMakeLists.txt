if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        RigidBody
        ParticleSystem
        Peridynamics
        SemiAnalyticalScheme
        Multiphysics
        FBXLoader
        UbiGUI
        ObjIO)
    add_example(Qt_AnimationCoupling Modeling LIB_DEPENDENCY)
endif()