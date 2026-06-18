if(PERIDYNO_LIBRARY_PARTICLESYSTEM AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        Modeling
        SemiAnalyticalScheme
        RigidBody
        Volume
        ObjIO
        QtGUI)
    add_example(SISPH_Turbing SemiAnalytical LIB_DEPENDENCY)

endif()

