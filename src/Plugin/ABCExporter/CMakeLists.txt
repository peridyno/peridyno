if(PERIDYNO_PLUGIN_ALEMBIC)
    set(LIB_DEPENDENCY Core Framework Topology Alembic)
    add_plugin(ABCExporter LIB_DEPENDENCY)
endif()