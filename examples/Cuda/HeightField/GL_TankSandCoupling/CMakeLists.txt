if(PERIDYNO_LIBRARY_HEIGHTFIELD)
    set(LIB_DEPENDENCY 
        HeightField
        Modeling
        GlfwGUI)
    add_example(GL_TankSandCoupling HeightField LIB_DEPENDENCY)
endif()