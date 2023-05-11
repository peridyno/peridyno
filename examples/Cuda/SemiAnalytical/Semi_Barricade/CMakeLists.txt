if(PERIDYNO_LIBRARY_SEMIANALYTICALSCHEME)
    set(LIB_DEPENDENCY 
        Core 
        Framework 
        IO 
        GLRenderEngine 
        SemiAnalyticalScheme 
        RigidBody 
        GlfwGUI)
    add_example(Semi_Barricade SemiAnalytical LIB_DEPENDENCY)
endif()