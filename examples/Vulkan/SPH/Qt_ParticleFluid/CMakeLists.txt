if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        ParticleSystem
        QtGUI)
    add_example(Qt_ParticleFluid SPH LIB_DEPENDENCY)
endif()