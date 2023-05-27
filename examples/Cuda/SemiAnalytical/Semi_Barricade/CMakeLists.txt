if(PERIDYNO_LIBRARY_SEMIANALYTICALSCHEME)
    set(LIB_DEPENDENCY 
        SemiAnalyticalScheme 
        GlfwGUI)
    add_example(Semi_Barricade SemiAnalytical LIB_DEPENDENCY)
endif()