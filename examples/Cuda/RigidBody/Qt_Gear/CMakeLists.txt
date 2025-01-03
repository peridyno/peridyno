if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        RigidBody
        ObjIO
        UbiGUI)
    add_example(Qt_Gear RigidBody LIB_DEPENDENCY)
endif()