option(PERIDYNO_PLUGIN_FBX "Enable binding the FBX library" OFF)

if(PERIDYNO_PLUGIN_FBX)
    set(LIB_DEPENDENCY Core Framework Topology GLRenderEngine IO )
    add_plugin(FBXLoader LIB_DEPENDENCY)
endif()
