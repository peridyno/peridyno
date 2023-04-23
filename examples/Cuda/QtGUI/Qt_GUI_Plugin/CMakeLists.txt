if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        Interaction
        ParticleSystem
        SemiAnalyticalScheme
        QtGUI)
    add_example(Qt_GUI_Plugin Qt LIB_DEPENDENCY)
endif()