if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        GlfwGUI)
    add_example(GL_HingeJoint RigidBody LIB_DEPENDENCY)
endif()