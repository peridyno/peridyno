if(PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        Modeling
        QtGUI)
    add_example(Qt_SAT RigidBody LIB_DEPENDENCY)
endif()