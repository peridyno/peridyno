if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY
        RigidBody
        ObjIO
        GlfwGUI)
    add_example(GL_ChainPyramidImpact RigidBody LIB_DEPENDENCY)
endif()
