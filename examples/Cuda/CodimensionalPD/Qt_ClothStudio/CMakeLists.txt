if(PERIDYNO_LIBRARY_PERIDYNAMICS AND PERIDYNO_QT_GUI)
    set(LIB_DEPENDENCY 
        Core 
        Framework 
        IO 
        Modeling
        Peridynamics
        Multiphysics
		QtGUI
        GLRenderEngine
        Interaction)
    add_example(Qt_ClothStudio CodimensionalPD LIB_DEPENDENCY)
endif()