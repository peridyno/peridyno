if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        Core 
        Framework 
        IO
        RigidBody 
        GlfwGUI
        GLRenderEngine)
    add_example(GL_CollisionDetectionIn3D RigidBody LIB_DEPENDENCY)
endif()
