if(PERIDYNO_LIBRARY_PERIDYNAMICS AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        Core 
        Framework 
        IO 
        Modeling
        Peridynamics
        Multiphysics
        GlfwGUI
        GLRenderEngine
        Interaction)
    add_example(GL_ClothWithCollision CodimensionalPD LIB_DEPENDENCY)
endif()