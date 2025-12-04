if(PERIDYNO_LIBRARY_VOLUME)
    set(LIB_DEPENDENCY
        Core 
        Framework 
        IO          
        Volume
        HeightField
        Modeling
        ABCExporter
        ObjIO
        QtGUI
        GLRenderEngine)
    add_example(Qt_AGrid2DBoat BalancedTrees LIB_DEPENDENCY)
endif()