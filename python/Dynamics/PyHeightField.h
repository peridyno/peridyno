#pragma once
#include "../PyCommon.h"

#include "HeightField/Module/ApplyBumpMap2TriangleSet.h"
#include "Module/TopologyMapping.h"
template <typename TDataType>
void declare_apply_bump_map_2_triangle_set(py::module& m, std::string typestr) {
	using Class = dyno::ApplyBumpMap2TriangleSet<TDataType>;
	using Parent = dyno::TopologyMapping;

	class ApplyBumpMap2TriangleSetTrampoline : public Class
	{
	public:
		using Class::Class;

		bool apply() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::ApplyBumpMap2TriangleSet<TDataType>,
				apply
				);
		}
	};

	class ApplyBumpMap2TriangleSetPublicist : public Class
	{
	public:
		using Class::apply;
	};

	std::string pyclass_name = std::string("ApplyBumpMap2TriangleSet") + typestr;
	py::class_<Class, Parent, ApplyBumpMap2TriangleSetTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("inHeightField", &Class::inHeightField, py::return_value_policy::reference)
		.def("outTriangleSet", &Class::outTriangleSet, py::return_value_policy::reference)
		.def("apply", &ApplyBumpMap2TriangleSetPublicist::apply, py::return_value_policy::reference);
}

#include "HeightField/Module/Steer.h"
template <typename TDataType>
void declare_steer(py::module& m, std::string typestr) {
	using Class = dyno::Steer<TDataType>;
	using Parent = dyno::KeyboardInputModule;

	class SteerTrampoline : public Class
	{
	public:
		using Class::Class;

		void onEvent(dyno::PKeyboardEvent event) override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Steer<TDataType>,
				onEvent,
				event
				);
		}
	};

	class SteerPublicist : public Class
	{
	public:
		using Class::onEvent;
	};

	std::string pyclass_name = std::string("Steer") + typestr;
	py::class_<Class, Parent, SteerTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varStrength", &Class::varStrength, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAngularVelocity", &Class::inAngularVelocity, py::return_value_policy::reference)
		.def("inQuaternion", &Class::inQuaternion, py::return_value_policy::reference)
		.def("onEvent", &SteerPublicist::onEvent, py::return_value_policy::reference);
}

#include "HeightField/CapillaryWave.h"
template <typename TDataType>
void declare_capillary_wave(py::module& m, std::string typestr) {
	using Class = dyno::CapillaryWave<TDataType>;
	using Parent = dyno::Node;

	class CapillaryWaveTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CapillaryWave<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CapillaryWave<TDataType>,
				updateStates
			);
		}
	};

	class CapillaryWavePublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;

		using Class::mDeviceGrid;
		using Class::mDeviceGridNext;
		using Class::mDeviceGridOld;
	};

	std::string pyclass_name = std::string("CapillaryWave") + typestr;
	py::class_<Class, Parent, CapillaryWaveTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varWaterLevel", &Class::varWaterLevel, py::return_value_policy::reference)
		.def("varResolution", &Class::varResolution, py::return_value_policy::reference)
		.def("varLength", &Class::varLength, py::return_value_policy::reference)
		.def("varViscosity", &Class::varViscosity, py::return_value_policy::reference)

		.def("stateHeight", &Class::stateHeight, py::return_value_policy::reference)
		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference)
		// public
		.def("setOriginX", &Class::setOriginX)
		.def("setOriginY", &Class::setOriginY)

		.def("getOriginX", &Class::getOriginX)
		.def("getOriginZ", &Class::getOriginZ)

		.def("getRealGridSize", &Class::getRealGridSize)
		.def("getOrigin", &Class::getOrigin)
		.def("moveDynamicRegion", &Class::moveDynamicRegion)
		// protected
		.def("resetStates", &CapillaryWavePublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &CapillaryWavePublicist::updateStates, py::return_value_policy::reference)

		.def_readwrite("mDeviceGrid", &CapillaryWavePublicist::mDeviceGrid)
		.def_readwrite("mDeviceGridNext", &CapillaryWavePublicist::mDeviceGridNext)
		.def_readwrite("mDeviceGridOld", &CapillaryWavePublicist::mDeviceGridOld);
}

#include "HeightField/GranularMedia.h"
template <typename TDataType>
void declare_granular_media(py::module& m, std::string typestr) {
	using Class = dyno::GranularMedia<TDataType>;
	using Parent = dyno::Node;

	class GranularMediaTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::GranularMedia<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::GranularMedia<TDataType>,
				updateStates
			);
		}
	};

	class GranularMediaPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("GranularMedia") + typestr;
	py::class_<Class, Parent, GranularMediaTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
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
		.def("stateInitialHeightField", &Class::stateInitialHeightField, py::return_value_policy::reference)
		// protected
		.def("resetStates", &GranularMediaPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &GranularMediaPublicist::updateStates, py::return_value_policy::reference);
}

#include "HeightField/LandScape.h"
template <typename TDataType>
void declare_land_scape(py::module& m, std::string typestr) {
	using Class = dyno::LandScape<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class LandScapeTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::LandScape<TDataType>,
				resetStates
			);
		}
	};

	class LandScapePublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::callbackTransform;
		using Class::callbackLoadFile;
	};

	std::string pyclass_name = std::string("LandScape") + typestr;
	py::class_<Class, Parent, LandScapeTrampoline,  std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varPatchSize", &Class::varPatchSize, py::return_value_policy::reference)
		.def("varFileName", &Class::varFileName, py::return_value_policy::reference)
		.def("stateInitialHeights", &Class::stateInitialHeights, py::return_value_policy::reference)
		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference)
		// protected
		.def("resetStates", &LandScapePublicist::resetStates, py::return_value_policy::reference)
		.def("callbackTransform", &LandScapePublicist::callbackTransform, py::return_value_policy::reference)
		.def("callbackLoadFile", &LandScapePublicist::callbackLoadFile, py::return_value_policy::reference);
}

#include "HeightField/OceanBase.h"
template <typename TDataType>
void declare_ocean_base(py::module& m, std::string typestr) {
	using Class = dyno::OceanBase<TDataType>;
	using Parent = dyno::Node;

	class OceanBaseTrampoline : public Class
	{
	public:
		using Class::Class;

		bool validateInputs() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::OceanBase<TDataType>,
				validateInputs
			);
		}
	};

	class OceanBasePublicist : public Class
	{
	public:
		using Class::validateInputs;
	};

	std::string pyclass_name = std::string("OceanBase") + typestr;
	py::class_<Class, Parent, OceanBaseTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("varWaterLevel", &Class::varWaterLevel, py::return_value_policy::reference)
		.def("getOceanPatch", &Class::getOceanPatch, py::return_value_policy::reference)
		.def("importOceanPatch", &Class::importOceanPatch, py::return_value_policy::reference)
		// protected
		.def("validateInputs", &OceanBasePublicist::validateInputs, py::return_value_policy::reference);
}

#include "HeightField/LargeOcean.h"
template <typename TDataType>
void declare_large_ocean(py::module& m, std::string typestr) {
	using Class = dyno::LargeOcean<TDataType>;
	using Parent = dyno::OceanBase < TDataType>;

	class LargeOceanTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::LargeOcean<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::LargeOcean<TDataType>,
				updateStates
			);
		}
	};

	class LargeOceanPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("LargeOcean") + typestr;
	py::class_<Class, Parent, LargeOceanTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varFileName", &Class::varFileName)
		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("stateTexCoord", &Class::stateTexCoord, py::return_value_policy::reference)
		.def("stateBumpMap", &Class::stateBumpMap, py::return_value_policy::reference)
		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference)
		// protected
		.def("resetStates", &LargeOceanPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &LargeOceanPublicist::updateStates, py::return_value_policy::reference);
}

#include "HeightField/MountainTorrents.h"
template <typename TDataType>
void declare_mountain_torrents(py::module& m, std::string typestr) {
	using Class = dyno::MountainTorrents<TDataType>;
	using Parent = dyno::CapillaryWave<TDataType>;

	class MountainTorrentsTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MountainTorrents<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MountainTorrents<TDataType>,
				updateStates
			);
		}
	};

	class MountainTorrentsPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("MountainTorrents") + typestr;
	py::class_<Class, Parent, MountainTorrentsTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getTerrain", &Class::getTerrain, py::return_value_policy::reference)
		.def("importTerrain", &Class::importTerrain, py::return_value_policy::reference)

		.def("stateInitialHeights", &Class::stateInitialHeights, py::return_value_policy::reference)
		// protected
		.def("resetStates", &MountainTorrentsPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &MountainTorrentsPublicist::updateStates, py::return_value_policy::reference);
}

#include "HeightField/Ocean.h"
template <typename TDataType>
void declare_ocean(py::module& m, std::string typestr) {
	using Class = dyno::Ocean<TDataType>;
	using Parent = dyno::OceanBase<TDataType>;

	class OceanTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Ocean<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Ocean<TDataType>,
				updateStates
			);
		}
	};

	class OceanPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("Ocean") + typestr;
	py::class_<Class, Parent, OceanTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varExtentX", &Class::varExtentX, py::return_value_policy::reference)
		.def("varExtentZ", &Class::varExtentZ, py::return_value_policy::reference)

		//DEF_NODE_PORTS
		.def("importCapillaryWaves", &Class::importCapillaryWaves, py::return_value_policy::reference)
		.def("getCapillaryWaves", &Class::getCapillaryWaves)
		.def("addCapillaryWave", &Class::addCapillaryWave)
		.def("removeCapillaryWave", &Class::removeCapillaryWave)
		//DEF_INSTANCE_STATE
		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference)
		// protected
		.def("resetStates", &OceanPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &OceanPublicist::updateStates, py::return_value_policy::reference);
}

#include "HeightField/OceanPatch.h"
template <typename TDataType>
void declare_ocean_patch(py::module& m, std::string typestr) {
	using Class = dyno::OceanPatch<TDataType>;
	using Parent = dyno::Node;

	class OceanPatchTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::OceanPatch<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::OceanPatch<TDataType>,
				updateStates
			);
		}

		void postUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::OceanPatch<TDataType>,
				postUpdateStates
			);
		}
	};

	class OceanPatchPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
		using Class::postUpdateStates;
	};

	std::string pyclass_name = std::string("OceanPatch") + typestr;
	py::class_<Class, Parent, OceanPatchTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
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

		.def("stateHeightField", &Class::stateHeightField, py::return_value_policy::reference)
		// protected
		.def("resetStates", &OceanPatchPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &OceanPatchPublicist::updateStates, py::return_value_policy::reference)
		.def("postUpdateStates", &OceanPatchPublicist::postUpdateStates, py::return_value_policy::reference);
}

#include "HeightField/RigidSandCoupling.h"
template <typename TDataType>
void declare_rigid_sand_coupling(py::module& m, std::string typestr) {
	using Class = dyno::RigidSandCoupling<TDataType>;
	using Parent = dyno::Node;

	class RigidSandCouplingTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::RigidSandCoupling<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::RigidSandCoupling<TDataType>,
				updateStates
			);
		}
	};

	class RigidSandCouplingPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("RigidSandCoupling") + typestr;
	py::class_<Class, Parent, RigidSandCouplingTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())

		//DEF_NODE_PORT
		.def("importGranularMedia", &Class::importGranularMedia, py::return_value_policy::reference)
		.def("getGranularMedia", &Class::getGranularMedia, py::return_value_policy::reference)
		//DEF_NODE_PORT
		.def("getRigidBodySystem", &Class::getRigidBodySystem, py::return_value_policy::reference)
		.def("importRigidBodySystem", &Class::importRigidBodySystem, py::return_value_policy::reference)
		// protected
		.def("resetStates", &RigidSandCouplingPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &RigidSandCouplingPublicist::updateStates, py::return_value_policy::reference);
}

#include "HeightField/RigidWaterCoupling.h"
template <typename TDataType>
void declare_rigid_water_coupling(py::module& m, std::string typestr) {
	using Class = dyno::RigidWaterCoupling<TDataType>;
	using Parent = dyno::Node;

	class RigidWaterCouplingTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::RigidWaterCoupling<TDataType>, 
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::RigidWaterCoupling<TDataType>,
				updateStates
			);
		}
	};

	class RigidWaterCouplingPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("RigidWaterCoupling") + typestr;
	py::class_<Class, Parent, RigidWaterCouplingTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
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
		.def("importOcean", &Class::importOcean, py::return_value_policy::reference)
		// protected
		.def("resetStates", &RigidWaterCouplingPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &RigidWaterCouplingPublicist::updateStates, py::return_value_policy::reference);
}

#include "HeightField/SurfaceParticleTracking.h"
template <typename TDataType>
void declare_surface_particle_tracking(py::module& m, std::string typestr) {
	using Class = dyno::SurfaceParticleTracking<TDataType>;
	using Parent = dyno::Node;

	class SurfaceParticleTrackingTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SurfaceParticleTracking<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SurfaceParticleTracking<TDataType>,
				updateStates
			);
		}

		bool validateInputs() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::SurfaceParticleTracking<TDataType>,
				validateInputs
			);
		}
	};

	class SurfaceParticleTrackingPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
		using Class::validateInputs;
	};

	std::string pyclass_name = std::string("SurfaceParticleTracking") + typestr;
	py::class_<Class, Parent, SurfaceParticleTrackingTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEV_VAR
		.def("varLayer", &Class::varLayer, py::return_value_policy::reference)
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)

		.def("getGranularMedia", &Class::getGranularMedia, py::return_value_policy::reference)
		.def("importGranularMedia", &Class::importGranularMedia, py::return_value_policy::reference)

		.def("statePointSet", &Class::statePointSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &SurfaceParticleTrackingPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &SurfaceParticleTrackingPublicist::updateStates, py::return_value_policy::reference)
		.def("validateInputs", &SurfaceParticleTrackingPublicist::validateInputs, py::return_value_policy::reference);
}

#include "HeightField/Vessel.h"
template <typename TDataType>
void declare_vessel(py::module& m, std::string typestr) {
	using Class = dyno::Vessel<TDataType>;
	using Parent = dyno::RigidBody<TDataType>;

	class VesselTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Vessel<TDataType>,
				resetStates
			);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Vessel<TDataType>,
				updateStates
			);
		}
	};

	class VesselPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("Vessel") + typestr;
	py::class_<Class, Parent, VesselTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
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
		.def("stateInstanceTransform", &Class::stateInstanceTransform, py::return_value_policy::reference)
		// protected
		.def("resetStates", &VesselPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &VesselPublicist::updateStates, py::return_value_policy::reference);
}

#include "HeightField/Wake.h"
template <typename TDataType>
void declare_wake(py::module& m, std::string typestr) {
	using Class = dyno::Wake<TDataType>;
	using Parent = dyno::CapillaryWave<TDataType>;

	class WakeTrampoline : public Class
	{
	public:
		using Class::Class;

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Wake<TDataType>,
				updateStates
			);
		}
	};

	class WakePublicist : public Class
	{
	public:
		using Class::updateStates;
	};

	std::string pyclass_name = std::string("Wake") + typestr;
	py::class_<Class, Parent, WakeTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEV_VAR
		.def("varMagnitude", &Class::varMagnitude, py::return_value_policy::reference)
		//DEF_NODE_PORT
		.def("getVessel", &Class::getVessel, py::return_value_policy::reference)
		.def("importVessel", &Class::importVessel, py::return_value_policy::reference)
		// protected
		.def("updateStates", &WakePublicist::updateStates, py::return_value_policy::reference);
}

//NumericalScheme

void pybind_height_field(py::module& m);