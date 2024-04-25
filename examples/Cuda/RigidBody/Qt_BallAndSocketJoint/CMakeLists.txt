if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        QtGUI)
    add_example(Qt_BallAndSocketJoint RigidBody LIB_DEPENDENCY)
endif()