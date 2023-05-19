if(PERIDYNO_LIBRARY_PERIDYNAMICS AND PERIDYNO_LIBRARY_MULTIPHYSICS)
    set(LIB_DEPENDENCY 
        GlfwGUI
        Modeling
        Multiphysics
        Peridynamics)
    add_example(GL_Cloth Peridynamics LIB_DEPENDENCY)
endif()