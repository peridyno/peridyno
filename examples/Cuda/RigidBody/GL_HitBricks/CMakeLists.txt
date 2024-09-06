if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        GlfwGUI)

    add_example(GL_HitBricks RigidBody LIB_DEPENDENCY)

    set_target_properties(GL_HitBricks PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_FLAGS})
endif()