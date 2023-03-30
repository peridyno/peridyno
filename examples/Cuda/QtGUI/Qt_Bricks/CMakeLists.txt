if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        QtGUI)
    add_example(Qt_Bricks Qt LIB_DEPENDENCY)
endif()