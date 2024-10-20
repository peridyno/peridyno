if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY 
        GlfwGUI
        QtGUI
        ParticleSystem)
    add_example(GL_GhostSPH SPH LIB_DEPENDENCY)
endif()