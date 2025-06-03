if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        GlfwGUI
        QtGUI
        SemiAnalyticalScheme
        Multiphysics
        QtGUI
        Volume
        ObjIO
        PartioExporter
        ParticleSystem)
    add_example(QT_ParticlesExport SPH LIB_DEPENDENCY)
endif()