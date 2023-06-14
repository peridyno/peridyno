if(PERIDYNO_LIBRARY_SEMIANALYTICALSCHEME)
    set(LIB_DEPENDENCY 
        Modeling
        SemiAnalyticalScheme 
        GlfwGUI)
    add_example(Semi_Barricade SemiAnalytical LIB_DEPENDENCY)
endif()