if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        Modeling
        SemiAnalyticalScheme 
        ObjIO
        ParticleSystem
        Core
        Volume
        QtGUI)
    add_example(SISPH_Pump SemiAnalytical LIB_DEPENDENCY)

endif()

