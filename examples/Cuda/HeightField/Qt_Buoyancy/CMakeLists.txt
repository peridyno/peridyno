if(PERIDYNO_Qt5_GUI AND PERIDYNO_LIBRARY_HEIGHTFIELD AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        HeightField
        RigidBody
        QtGUI)
    add_example(Qt_Buoyancy HeightField LIB_DEPENDENCY)
endif()