if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_QT_GUI)
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