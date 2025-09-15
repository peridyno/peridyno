option(PERIDYNO_PLUGIN_STLIO "Enable binding the stl model library" ON)

if(PERIDYNO_PLUGIN_STLIO)
    set(LIB_DEPENDENCY Core Framework IO Topology GLRenderEngine)
    add_plugin(STLIO LIB_DEPENDENCY)
endif()
