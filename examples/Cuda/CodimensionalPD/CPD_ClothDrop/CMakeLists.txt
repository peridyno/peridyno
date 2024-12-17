if(PERIDYNO_LIBRARY_PERIDYNAMICS)
    set(LIB_DEPENDENCY 
        Modeling
        Peridynamics
        Multiphysics
	    UbiGUI)
    add_example(CPD_ClothDrop CodimensionalPD LIB_DEPENDENCY)
endif()