if(PERIDYNO_LIBRARY_HEIGHTFIELD)
    set(LIB_DEPENDENCY 
        HeightField
        GlfwGUI)
    add_example(GL_Ocean HeightField LIB_DEPENDENCY)
endif()