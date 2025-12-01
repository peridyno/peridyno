if(PERIDYNO_LIBRARY_VOLUME)
    set(LIB_DEPENDENCY 
        ABCExporter
        Core 
        Framework 
        IO          
        Volume
        Modeling
        ParticleSystem
        UbiGUI
        GLRenderEngine
        EulerFluid)
    add_example(GL_AGridQuicksandInlet AASDF LIB_DEPENDENCY)
endif()