if(PERIDYNO_LIBRARY_SEMIANALYTICALSCHEME)
    set(LIB_DEPENDENCY 
        Core 
        Framework 
        IO 
        Modeling
        GLRenderEngine 
        SemiAnalyticalScheme 
        RigidBody 
        UbiGUI)
    add_example(Semi_Bar SemiAnalytical LIB_DEPENDENCY)
endif()