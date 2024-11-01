if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        Modeling
        QtGUI
        SkeletonLoader
        ObjIO)
    add_example(Qt_ConfigurationQuadruped RigidBody LIB_DEPENDENCY)
endif()