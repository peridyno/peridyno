if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY 
        ParticleSystem
        QtGUI)
    add_example(Qt_ParticleFluid SPH LIB_DEPENDENCY)
endif()