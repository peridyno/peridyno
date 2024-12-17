if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        UbiGUI)
    add_example(GL_RigidCompound RigidBody LIB_DEPENDENCY)
endif()