if(PERIDYNO_LIBRARY_VOLUME)
    set(LIB_DEPENDENCY        
        Volume
        UbiGUI
        GLRenderEngine)
    add_example(GL_SDFUniformBoolean Volume LIB_DEPENDENCY)
endif()
