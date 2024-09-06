if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        Modeling
        ParticleSystem
        QtGUI)
    add_example(GL_SemiImplicitDensitySolver SPH LIB_DEPENDENCY)
endif()