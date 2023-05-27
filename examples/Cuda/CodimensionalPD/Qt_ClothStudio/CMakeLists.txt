if(PERIDYNO_LIBRARY_PERIDYNAMICS AND PERIDYNO_Qt5_GUI)
    set(LIB_DEPENDENCY 
        Core 
        Framework 
        IO 
        Peridynamics
        Multiphysics
		QtGUI
        GLRenderEngine
        Interaction)
    add_example(Qt_ClothStudio CodimensionalPD LIB_DEPENDENCY)
endif()