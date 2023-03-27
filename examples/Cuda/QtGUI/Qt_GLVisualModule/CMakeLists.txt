if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        ParticleSystem
        Modeling
        QtGUI)
    add_example(Qt_GLVisualModule Qt LIB_DEPENDENCY)
endif()

