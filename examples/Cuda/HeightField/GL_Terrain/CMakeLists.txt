if(PERIDYNO_LIBRARY_HEIGHTFIELD)
    set(LIB_DEPENDENCY 
        HeightField
        GlfwGUI)
    add_example(GL_Terrain HeightField LIB_DEPENDENCY)
endif()