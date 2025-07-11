if(PERIDYNO_LIBRARY_HEIGHTFIELD)
    set(LIB_DEPENDENCY 
        HeightField
        UbiGUI)
    add_example(GL_MoutainTorrents HeightField LIB_DEPENDENCY)
endif()