if(PERIDYNO_LIBRARY_HEIGHTFIELD)
    set(LIB_DEPENDENCY 
        HeightField
        Modeling
        UbiGUI)
    add_example(GL_RigidSandCoupling HeightField LIB_DEPENDENCY)
endif()