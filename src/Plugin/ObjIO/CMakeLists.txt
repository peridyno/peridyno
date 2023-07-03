option(PERIDYNO_PLUGIN_OBJ "Enable binding the obj library" OFF)

if(PERIDYNO_PLUGIN_OBJ)
    set(LIB_DEPENDENCY Core Framework Topology GLRenderEngine)
    add_plugin(ObjIO LIB_DEPENDENCY)
endif()
