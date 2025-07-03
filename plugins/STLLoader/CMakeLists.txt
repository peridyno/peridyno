option(PERIDYNO_PLUGIN_STLLOADER "Enable binding the stl model library" ON)

if(PERIDYNO_PLUGIN_STL3D)
    set(LIB_DEPENDENCY Core Framework IO Topology GLRenderEngine)
    add_plugin(StlLoaer LIB_DEPENDENCY)
endif()
