if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_Qt5_GUI)
    set(LIB_DEPENDENCY 
        Modeling
        SemiAnalyticalScheme 
        QtGUI)
    add_example(Qt_WaterPouring SemiAnalytical LIB_DEPENDENCY)
endif()

