#pragma once
#include "../PyCommon.h"

#include "HeightField/OceanPatch.h"
template <typename TDataType>
void declare_ocean_patch(py::module& m, std::string typestr) {
	using Class = dyno::OceanPatch<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("OceanPatch") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_wind_type", &Class::varWindType, py::return_value_policy::reference)
		.def("var_amplitude", &Class::varAmplitude, py::return_value_policy::reference)
		.def("var_wind_speed", &Class::varWindSpeed, py::return_value_policy::reference)
		.def("var_choppiness", &Class::varChoppiness, py::return_value_policy::reference)
		.def("var_global_shift", &Class::varGlobalShift, py::return_value_policy::reference)
		.def("var_wind_direction", &Class::varWindDirection, py::return_value_policy::reference)
		.def("var_resolution", &Class::varResolution, py::return_value_policy::reference)
		.def("var_patch_size", &Class::varPatchSize, py::return_value_policy::reference)
		.def("var_time_scale", &Class::varTimeScale, py::return_value_policy::reference)
		.def("state_displacement", &Class::stateDisplacement, py::return_value_policy::reference)
		.def("state_height_field", &Class::stateHeightField, py::return_value_policy::reference);
}

#include "HeightField/Ocean.h"
template <typename TDataType>
void declare_ocean(py::module& m, std::string typestr) {
	using Class = dyno::Ocean<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Ocean") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_extentX", &Class::varExtentX, py::return_value_policy::reference)
		.def("var_extentZ", &Class::varExtentZ, py::return_value_policy::reference)
		//DEF_NODE_PORT
		.def("get_ocean_patch", &Class::getOceanPatch)
		.def("import_ocean_patch", &Class::importOceanPatch, py::return_value_policy::reference)
		//DEF_NODE_PORTS
		.def("import_capillary_waves", &Class::importCapillaryWaves, py::return_value_policy::reference)
		.def("get_capillary_waves", &Class::getCapillaryWaves)
		.def("add_capillary_wave", &Class::addCapillaryWave)
		.def("remove_capillary_wave", &Class::removeCapillaryWave)
		//DEF_INSTANCE_STATE
		.def("state_height_field", &Class::stateHeightField, py::return_value_policy::reference);
}

#include "HeightField/CapillaryWave.h"
template <typename TDataType>
void declare_capillary_wave(py::module& m, std::string typestr) {
	using Class = dyno::CapillaryWave<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("CapillaryWave") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_water_level", &Class::varWaterLevel, py::return_value_policy::reference)
		.def("var_resolution", &Class::varResolution, py::return_value_policy::reference)
		.def("var_length", &Class::varLength, py::return_value_policy::reference)
		.def("state_height", &Class::stateHeight, py::return_value_policy::reference)
		.def("state_height_field", &Class::stateHeightField, py::return_value_policy::reference)
		//public
		.def("set_originX", &Class::setOriginX)
		.def("set_originY", &Class::setOriginY)
		.def("get_originX", &Class::getOriginX)
		.def("get_originZ", &Class::getOriginZ)
		.def("get_real_grid_size", &Class::getRealGridSize)
		.def("get_origin", &Class::getOrigin)
		.def("move_dynamic_region", &Class::moveDynamicRegion);
}

#include "HeightField/Coupling.h"
template <typename TDataType>
void declare_coupling(py::module& m, std::string typestr) {
	using Class = dyno::Coupling<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("Coupling") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("var_damping", &Class::varDamping, py::return_value_policy::reference)
		.def("var_rotational_damping", &Class::varRotationalDamping, py::return_value_policy::reference)
		//DEF_NODE_PORTS
		.def("import_vessels", &Class::importVessels, py::return_value_policy::reference)
		.def("get_vessels", &Class::getVessels, py::return_value_policy::reference)
		.def("add_vessel", &Class::addVessel)
		.def("remove_vessel", &Class::removeVessel)
		//DEF_NODE_PORT
		.def("get_ocean", &Class::getOcean)
		.def("import_ocean", &Class::importOcean, py::return_value_policy::reference);
}

#include "HeightField/GranularMedia.h"
template <typename TDataType>
void declare_granular_media(py::module& m, std::string typestr) {
	using Class = dyno::GranularMedia<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("GranularMedia") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEV_VAR
		.def("var_width", &Class::varWidth, py::return_value_policy::reference)
		.def("var_height", &Class::varHeight, py::return_value_policy::reference)
		.def("var_depth", &Class::varDepth, py::return_value_policy::reference)
		.def("var_coefficient_of_drag_force", &Class::varCoefficientOfDragForce, py::return_value_policy::reference)
		.def("var_coefficient_of_friction", &Class::varCoefficientOfFriction, py::return_value_policy::reference)
		.def("var_spacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("var_gravity", &Class::varGravity, py::return_value_policy::reference)
		.def("state_land_scape", &Class::stateLandScape, py::return_value_policy::reference)
		.def("state_grid", &Class::stateGrid, py::return_value_policy::reference)
		.def("state_grid_next", &Class::stateGridNext, py::return_value_policy::reference)
		.def("state_height_field", &Class::stateHeightField, py::return_value_policy::reference);
}

#include "HeightField/LandScape.h"
template <typename TDataType>
void declare_land_scape(py::module& m, std::string typestr) {
	using Class = dyno::LandScape<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("LandScape") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEV_VAR
		.def("var_patch_size", &Class::varPatchSize, py::return_value_policy::reference)
		.def("var_file_name", &Class::varFileName, py::return_value_policy::reference)
		.def("state_height_field", &Class::stateHeightField, py::return_value_policy::reference);
}

#include "HeightField/SurfaceParticleTracking.h"
template <typename TDataType>
void declare_surface_particle_tracking(py::module& m, std::string typestr) {
	using Class = dyno::SurfaceParticleTracking<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("SurfaceParticleTracking") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEV_VAR
		.def("var_layer", &Class::varLayer, py::return_value_policy::reference)
		.def("var_spacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("get_granular_media", &Class::getGranularMedia)
		.def("import_granular_media", &Class::importGranularMedia, py::return_value_policy::reference)
		.def("state_point_set", &Class::statePointSet, py::return_value_policy::reference);
}

#include "HeightField/Vessel.h"
template <typename TDataType>
void declare_vessel(py::module& m, std::string typestr) {
	using Class = dyno::Vessel<TDataType>;
	using Parent = dyno::RigidBody<TDataType>;
	std::string pyclass_name = std::string("Vessel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("bounding_box", &Class::boundingBox)
		//DEV_VAR
		.def("var_barycenter_offset", &Class::varBarycenterOffset, py::return_value_policy::reference)
		.def("var_envelope_name", &Class::varEnvelopeName, py::return_value_policy::reference)
		.def("var_test", &Class::varTest, py::return_value_policy::reference)
		.def("var_density", &Class::varDensity, py::return_value_policy::reference)
		.def("state_barycenter", &Class::stateBarycenter, py::return_value_policy::reference)
		.def("state_envelope", &Class::stateEnvelope, py::return_value_policy::reference)
		.def("state_mesh", &Class::stateMesh, py::return_value_policy::reference)
		.def("in_texture_mesh", &Class::inTextureMesh, py::return_value_policy::reference)
		.def("state_instance_transform", &Class::stateInstanceTransform, py::return_value_policy::reference);
}

#include "HeightField/Wake.h"
template <typename TDataType>
void declare_wake(py::module& m, std::string typestr) {
	using Class = dyno::Wake<TDataType>;
	using Parent = dyno::CapillaryWave<TDataType>;
	std::string pyclass_name = std::string("Wake") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEV_VAR
		.def("var_magnitude", &Class::varMagnitude, py::return_value_policy::reference)
		//DEF_NODE_PORT
		.def("get_vessel", &Class::getVessel)
		.def("import_vessel", &Class::importVessel, py::return_value_policy::reference);
}

#include "HeightField/Module/Steer.h"
template <typename TDataType>
void declare_steer(py::module& m, std::string typestr) {
	using Class = dyno::Steer<TDataType>;
	using Parent = dyno::KeyboardInputModule;
	std::string pyclass_name = std::string("Steer") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEV_VAR
		.def("in_velocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("in_angular_velocity", &Class::inAngularVelocity, py::return_value_policy::reference)
		.def("in_quaternion", &Class::inQuaternion, py::return_value_policy::reference);
}

//void declare_height_field_initializer(py::module& m, std::string typestr);

void pybind_height_field(py::module& m);