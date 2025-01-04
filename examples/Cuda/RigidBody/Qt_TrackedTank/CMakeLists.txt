if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        RigidBody
        UbiGUI)
    add_example(Qt_TrackedTank RigidBody LIB_DEPENDENCY)
endif()