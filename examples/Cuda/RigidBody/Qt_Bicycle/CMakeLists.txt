if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        Modeling
        QtGUI
        ObjIO)
    add_example(Qt_Bicycle RigidBody LIB_DEPENDENCY)
endif()