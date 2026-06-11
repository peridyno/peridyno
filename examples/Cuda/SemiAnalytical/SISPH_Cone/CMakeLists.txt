if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        Modeling
        SemiAnalyticalScheme 
        QtGUI
        ObjIO)
    add_example(SISPH_Cone SemiAnalytical LIB_DEPENDENCY)

endif()

