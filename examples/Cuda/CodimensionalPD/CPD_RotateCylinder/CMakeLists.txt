if(PERIDYNO_LIBRARY_PERIDYNAMICS)
    set(LIB_DEPENDENCY 
        Modeling
        Peridynamics
        Multiphysics
	    UbiGUI)
    add_example(CPD_RotateCylinder CodimensionalPD LIB_DEPENDENCY)
endif()