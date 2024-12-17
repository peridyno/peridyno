if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY 
        GlfwGUI
        ParticleSystem
        Multiphysics)
    add_example(GL_ParticleEmitter SPH LIB_DEPENDENCY)
endif()