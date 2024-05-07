if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        QtGUI)
    add_example(GL_FixedJoint RigidBody LIB_DEPENDENCY)
endif()