if(PERIDYNO_LIBRARY_PERIDYNAMICS)
    set(LIB_DEPENDENCY 
        UbiGUI
        Multiphysics
        Peridynamics)
    add_example(GL_Plasticity Peridynamics LIB_DEPENDENCY)
endif()