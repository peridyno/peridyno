if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY
        Modeling
        Interaction
        ParticleSystem
        QtGUI)
    add_example(Qt_GUI_Plugin Qt LIB_DEPENDENCY)
endif()