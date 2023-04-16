if(PERIDYNO_Qt5_GUI AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        QtGUI)
    add_example(Qt_LinearBVH RigidBody LIB_DEPENDENCY)
endif()