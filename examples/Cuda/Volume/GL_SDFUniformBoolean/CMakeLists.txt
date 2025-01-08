if(PERIDYNO_LIBRARY_VOLUME)
    set(LIB_DEPENDENCY 
        Core 
        Framework 
        IO          
        Volume
        ABCExporter
		ObjIO
        UbiGUI
        GLRenderEngine)
    add_example(GL_SDFUniformBoolean Volume LIB_DEPENDENCY)
endif()
