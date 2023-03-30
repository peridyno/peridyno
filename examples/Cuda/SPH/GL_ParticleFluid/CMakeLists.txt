if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY 
        Modeling
        ParticleSystem
        GlfwGUI)
    add_example(GL_ParticleFluid SPH LIB_DEPENDENCY)
endif()