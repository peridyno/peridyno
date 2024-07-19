if(PERIDYNO_QT_GUI AND PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY 
        RigidBody
        Modeling
        GlfwGUI)
    add_example(Qt_Vechicle RigidBody LIB_DEPENDENCY)
endif()