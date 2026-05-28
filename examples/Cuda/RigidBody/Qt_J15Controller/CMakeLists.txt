if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        Modeling
        QtGUI
        ObjIO
        GlfwGUI
    )
    add_example(Qt_J15Controller RigidBody LIB_DEPENDENCY)
endif()