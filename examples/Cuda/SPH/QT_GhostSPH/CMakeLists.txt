if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY 
        GlfwGUI
        QtGUI
        SemiAnalyticalScheme
        Multiphysics
        QtGUI
        Volume
        ObjIO
        ParticleSystem)
    add_example(QT_GhostSPH SPH LIB_DEPENDENCY)
endif()