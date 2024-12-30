if(PERIDYNO_LIBRARY_VOLUME)
    set(LIB_DEPENDENCY        
        Volume
        UbiGUI)
    add_example(GL_SDFUniform AASDF LIB_DEPENDENCY)
endif()
