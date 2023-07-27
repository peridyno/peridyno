if(PERIDYNO_LIBRARY_SEMIANALYTICALSCHEME)
    set(LIB_DEPENDENCY 
        Core 
        Framework 
        IO 
        Modeling
        GLRenderEngine 
        SemiAnalyticalScheme 
        RigidBody 
        GlfwGUI)
    add_example(Semi_Bar SemiAnalytical LIB_DEPENDENCY)
endif()