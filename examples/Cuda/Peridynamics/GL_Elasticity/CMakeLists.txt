if(PERIDYNO_LIBRARY_PERIDYNAMICS)
    set(LIB_DEPENDENCY 
        GlfwGUI
        Peridynamics)
    add_example(GL_Elasticity Peridynamics LIB_DEPENDENCY)
endif()