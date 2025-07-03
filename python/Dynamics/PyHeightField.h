#pragma once
#include "../PyCommon.h"

#include "HeightField/Module/ApplyBumpMap2TriangleSet.h"
#include "Module/TopologyMapping.h"
template <typename TDataType>
void declare_apply_bump_map_2_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::ApplyBumpMap2TriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;
	std::string pyclass_name = std::string("ApplyBumpMap2TriangleSet") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("inHeightField", &Class::inHeightField, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference);
}

#include "HeightField/Module/Steer.h"
template <typename TDataType>
void declare_steer(py::module& m, std::string typestr) {
	using Class = dyno::Steer<TDataType>;
	using Parent = dyno::KeyboardInputModule;
	std::string pyclass_name = std::string("Steer") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varStrength", &Class::varStrength, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAngularVelocity", &Class::inAngularVelocity, py::return_value_policy::reference)
		.def("inQuaternion", &Class::inQuaternion, py::return_value_policy::reference);
}

#include "HeightField/CapillaryWave.h"
template <typename TDataType>
void declare_capillary_wave(py::module& m, std::string typestr) {
	using Class = dyno::CapillaryWave<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("CapillaryWave") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varWaterLevel", &Class::varWaterLevel, py::return_value_policy::reference)
		.def("varResolution", &Class::varResolution, py::return_value_policy::reference)
		.def("varLength", &Class::varLength, py::return_value_policy::reference)
		.def("varViscosity", &Class::varViscosity, py::return_value_policy::reference)
		.def("stateHeight", &Class::stateHeight, py::return_value_policy::reference)
		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference)
		//public
		.def("setOriginX", &Class::setOriginX)
		.def("setOriginY", &Class::setOriginY)
		.def("getOriginX", &Class::getOriginX)
		.def("getOriginZ", &Class::getOriginZ)
		.def("getRealGridSize", &Class::getRealGridSize)
		.def("getOrigin", &Class::getOrigin)
		.def("moveDynamicRegion", &Class::moveDynamicRegion);
}

#include "HeightField/GranularMedia.h"
template <typename TDataType>
void declare_granular_media(py::module& m, std::string typestr) {
	using Class = dyno::GranularMedia<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("GranularMedia") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varOrigin", &Class::varOrigin, py::return_value_policy::reference)
		.def("varWidth", &Class::varWidth, py::return_value_policy::reference)
		.def("varHeight", &Class::varHeight, py::return_value_policy::reference)
		.def("varDepth", &Class::varDepth, py::return_value_policy::reference)
		.def("varDepthOfDiluteLayer", &Class::varDepthOfDiluteLayer, py::return_value_policy::reference)
		.def("varCoefficientOfDragForce", &Class::varCoefficientOfDragForce, py::return_value_policy::reference)
		.def("varCoefficientOfFriction", &Class::varCoefficientOfFriction, py::return_value_policy::reference)
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("varGravity", &Class::varGravity, py::return_value_policy::reference)
		.def("stateLandScape", &Class::stateLandScape, py::return_value_policy::reference)
		.def("stateGrid", &Class::stateGrid, py::return_value_policy::reference)
		.def("stateGridNext", &Class::stateGridNext, py::return_value_policy::reference)
		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference)
		.def("stateInitialHeightField", &Class::stateInitialHeightField, py::return_value_policy::reference);
}

#include "HeightField/LandScape.h"
template <typename TDataType>
void declare_land_scape(py::module& m, std::string typestr) {
	using Class = dyno::LandScape<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;
	std::string pyclass_name = std::string("LandScape") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getInitialHeights", &Class::getInitialHeights)
		//DEV_VAR
		.def("varPatchSize", &Class::varPatchSize, py::return_value_policy::reference)
		.def("varFileName", &Class::varFileName, py::return_value_policy::reference)
		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference);
}

#include "HeightField/OceanBase.h"
template <typename TDataType>
void declare_ocean_base(py::module& m, std::string typestr) {
	using Class = dyno::OceanBase<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("OceanBase") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("varWaterLevel", &Class::varWaterLevel, py::return_value_policy::reference)
		.def("getOceanPatch", &Class::getOceanPatch, py::return_value_policy::reference)
		.def("importOceanPatch", &Class::importOceanPatch, py::return_value_policy::reference);
}

#include "HeightField/LargeOcean.h"
template <typename TDataType>
void declare_large_ocean(py::module& m, std::string typestr) {
	using Class = dyno::LargeOcean<TDataType>;
	using Parent = dyno::OceanBase < TDataType>;
	std::string pyclass_name = std::string("LargeOcean") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFileName", &Class::varFileName)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("stateTexCoord", &Class::stateTexCoord, py::return_value_policy::reference)
		.def("stateBumpMap", &Class::stateBumpMap, py::return_value_policy::reference)
		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference);
}

#include "HeightField/Ocean.h"
template <typename TDataType>
void declare_ocean(py::module& m, std::string typestr) {
	using Class = dyno::Ocean<TDataType>;
	using Parent = dyno::OceanBase<TDataType>;
	std::string pyclass_name = std::string("Ocean") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varExtentX", &Class::varExtentX, py::return_value_policy::reference)
		.def("varExtentZ", &Class::varExtentZ, py::return_value_policy::reference)
		.def("varWaterLevel", &Class::varWaterLevel, py::return_value_policy::reference)
		//DEF_NODE_PORTS
		.def("importCapillaryWaves", &Class::importCapillaryWaves, py::return_value_policy::reference)
		.def("getCapillaryWaves", &Class::getCapillaryWaves)
		.def("addCapillaryWave", &Class::addCapillaryWave)
		.def("removeCapillaryWave", &Class::removeCapillaryWave)
		//DEF_INSTANCE_STATE
		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference);
}

#include "HeightField/OceanPatch.h"
template <typename TDataType>
void declare_ocean_patch(py::module& m, std::string typestr) {
	using Class = dyno::OceanPatch<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("OceanPatch") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("varWindType", &Class::varWindType, py::return_value_policy::reference)
		.def("varAmplitude", &Class::varAmplitude, py::return_value_policy::reference)
		.def("varAmplitudeScale", &Class::varAmplitudeScale, py::return_value_policy::reference)
		.def("varWindSpeed", &Class::varWindSpeed, py::return_value_policy::reference)
		.def("varChoppiness", &Class::varChoppiness, py::return_value_policy::reference)
		.def("varGlobalShift", &Class::varGlobalShift, py::return_value_policy::reference)

		.def("varWindDirection", &Class::varWindDirection, py::return_value_policy::reference)
		.def("varResolution", &Class::varResolution, py::return_value_policy::reference)
		.def("varPatchSize", &Class::varPatchSize, py::return_value_policy::reference)
		.def("varTimeScale", &Class::varTimeScale, py::return_value_policy::reference)

		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference);
}

#include "HeightField/RigidSandCoupling.h"
template <typename TDataType>
void declare_rigid_sand_coupling(py::module& m, std::string typestr) {
	using Class = dyno::RigidSandCoupling<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("RigidSandCoupling") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())

		//DEF_NODE_PORT
		.def("importGranularMedia", &Class::importGranularMedia, py::return_value_policy::reference)
		.def("getGranularMedia", &Class::getGranularMedia, py::return_value_policy::reference)
		//DEF_NODE_PORT
		.def("getRigidBodySystem", &Class::getRigidBodySystem, py::return_value_policy::reference)
		.def("importRigidBodySystem", &Class::importRigidBodySystem, py::return_value_policy::reference);
}

#include "HeightField/RigidWaterCoupling.h"
template <typename TDataType>
void declare_rigid_water_coupling(py::module& m, std::string typestr) {
	using Class = dyno::RigidWaterCoupling<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("RigidWaterCoupling") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varDamping", &Class::varDamping, py::return_value_policy::reference)
		.def("varRotationalDamping", &Class::varRotationalDamping, py::return_value_policy::reference)
		//DEF_NODE_PORTS
		.def("importVessels", &Class::importVessels, py::return_value_policy::reference)
		.def("getVessels", &Class::getVessels)
		.def("addVessel", &Class::addVessel)
		.def("removeVessel", &Class::removeVessel)
		//DEF_NODE_PORT
		.def("getOcean", &Class::getOcean, py::return_value_policy::reference)
		.def("importOcean", &Class::importOcean, py::return_value_policy::reference);
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
		.def("varLayer", &Class::varLayer, py::return_value_policy::reference)
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		.def("getGranularMedia", &Class::getGranularMedia, py::return_value_policy::reference)
		.def("importGranularMedia", &Class::importGranularMedia, py::return_value_policy::reference)
		.def("statePointSet", &Class::statePointSet, py::return_value_policy::reference);
}

#include "HeightField/Vessel.h"
template <typename TDataType>
void declare_vessel(py::module& m, std::string typestr) {
	using Class = dyno::Vessel<TDataType>;
	using Parent = dyno::RigidBody<TDataType>;
	std::string pyclass_name = std::string("Vessel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("boundingBox", &Class::boundingBox)
		.def("getNodeType", &Class::getNodeType)
		//DEV_VAR
		.def("varBarycenterOffset", &Class::varBarycenterOffset, py::return_value_policy::reference)
		.def("varEnvelopeName", &Class::varEnvelopeName, py::return_value_policy::reference)
		.def("varTextureMeshName", &Class::varTextureMeshName, py::return_value_policy::reference)
		.def("varDensity", &Class::varDensity, py::return_value_policy::reference)
		.def("varInitialMass", &Class::varInitialMass, py::return_value_policy::reference)
		.def("stateBarycenter", &Class::stateBarycenter, py::return_value_policy::reference)
		.def("stateEnvelope", &Class::stateEnvelope, py::return_value_policy::reference)
		.def("stateTextureMesh", &Class::stateTextureMesh, py::return_value_policy::reference)
		.def("stateInstanceTransform", &Class::stateInstanceTransform, py::return_value_policy::reference);
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
		.def("varMagnitude", &Class::varMagnitude, py::return_value_policy::reference)
		//DEF_NODE_PORT
		.def("getVessel", &Class::getVessel, py::return_value_policy::reference)
		.def("importVessel", &Class::importVessel, py::return_value_policy::reference);
}

//NumericalScheme

void pybind_height_field(py::module& m);