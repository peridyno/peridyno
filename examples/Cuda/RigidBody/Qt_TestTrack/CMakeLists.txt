if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        Modeling
        QtGUI)
    add_example(Qt_TestTrack RigidBody LIB_DEPENDENCY)
endif()