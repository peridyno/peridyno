option(PERIDYNO_PLUGIN_FBX "Enable binding the FBX library" OFF)

if(PERIDYNO_PLUGIN_FBX)
    set(LIB_DEPENDENCY Core Framework Topology)
    add_plugin(SkeletonLoader LIB_DEPENDENCY)
endif()
