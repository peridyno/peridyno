if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        Modeling
        SemiAnalyticalScheme 
        QtGUI)
    add_example(Qt_WaterPouring SemiAnalytical LIB_DEPENDENCY)
endif()

