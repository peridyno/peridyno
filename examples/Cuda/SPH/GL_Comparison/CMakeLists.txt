if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY 
        ParticleSystem
        Multiphysics
        UbiGUI)
    add_example(GL_Comparison SPH LIB_DEPENDENCY)
endif()