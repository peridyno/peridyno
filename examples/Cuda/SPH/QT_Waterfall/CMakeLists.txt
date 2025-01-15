if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        Modeling
        ParticleSystem
        IO 
        SemiAnalyticalScheme
        ObjIO
        ABCExporter
        QtGUI)
    add_example(QT_Waterfall SPH LIB_DEPENDENCY)
endif()