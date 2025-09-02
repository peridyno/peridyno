if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY 
        Modeling
        SemiAnalyticalScheme 
        ObjIO
        ParticleSystem
        Core
         UbiGUI)
    add_example(Qt_WaterPouring SemiAnalytical LIB_DEPENDENCY)
endif()

