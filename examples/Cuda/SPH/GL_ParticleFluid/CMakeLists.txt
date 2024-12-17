if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY 
        ParticleSystem
        Multiphysics
        UbiGUI)
    add_example(GL_ParticleFluid SPH LIB_DEPENDENCY)
endif()