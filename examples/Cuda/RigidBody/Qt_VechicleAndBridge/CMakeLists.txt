if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        Modeling
        UbiGUI
	ObjIO)
    add_example(Qt_VechicleAndBridge RigidBody LIB_DEPENDENCY)
endif()