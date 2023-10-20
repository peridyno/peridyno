if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY 
        GlfwGUI
        ParticleSystem
        ImWidgets
        QtGUI)
    add_example(Qt_ImChart Qt LIB_DEPENDENCY)
endif()