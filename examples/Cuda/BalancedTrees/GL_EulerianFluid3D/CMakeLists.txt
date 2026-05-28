if(PERIDYNO_LIBRARY_VOLUME)
    set(LIB_DEPENDENCY 
        UbiGUI
        EulerFluid)
    add_example(GL_EulerianFluid3D BalancedTrees LIB_DEPENDENCY)
endif()