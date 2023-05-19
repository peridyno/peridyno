if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        QtGUI)
    add_example(Qt_Bricks RigidBody LIB_DEPENDENCY)
endif()