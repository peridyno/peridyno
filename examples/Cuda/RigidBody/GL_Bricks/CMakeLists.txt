if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        UbiGUI)
    add_example(GL_Bricks RigidBody LIB_DEPENDENCY)
endif()