if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        Modeling
        QtGUI
        HeightField
        FBXLoader
        MujocoLoader
        STLIO
        )
    add_example(Qt_MujocoXML RigidBody LIB_DEPENDENCY)
endif()