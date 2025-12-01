if(PERIDYNO_LIBRARY_VOLUME)
    set(LIB_DEPENDENCY
        ABCExporter
        Volume
        ParticleSystem
        UbiGUI
        EulerFluid)
    add_example(GL_AGridQuicksandEmitter AASDF LIB_DEPENDENCY)
endif()