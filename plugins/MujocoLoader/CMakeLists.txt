option(PERIDYNO_PLUGIN_MUJOCO"Enable binding the MUJOCO library" OFF)

if(PERIDYNO_PLUGIN_FBX)
    set(LIB_DEPENDENCY Core Framework Topology GLRenderEngine IO ObjIO FBXLoader Modeling RigidBody Multiphysics)
    add_plugin(MujocoLoader LIB_DEPENDENCY)
endif()
