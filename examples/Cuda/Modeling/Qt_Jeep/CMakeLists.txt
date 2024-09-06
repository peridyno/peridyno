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
    add_example(Qt_GUI_Jeep Modeling LIB_DEPENDENCY)
endif()