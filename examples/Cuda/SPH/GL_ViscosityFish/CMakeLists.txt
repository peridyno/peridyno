if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY 
        Modeling
        ParticleSystem
        Multiphysics
        GlfwGUI)
    add_example(GL_ViscosityFish SPH LIB_DEPENDENCY)
endif()