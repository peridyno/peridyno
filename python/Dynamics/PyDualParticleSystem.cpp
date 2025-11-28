#pragma once
#include "PyDualParticleSystem.h"

#include "DualParticleSystem/Module/VirtualFissionFusionStrategy.h"
template <typename TDataType>
void declare_virtual_fission_fusion_strategy(py::module& m, std::string typestr) {
	using Class = dyno::VirtualFissionFusionStrategy<TDataType>;
	using Parent = dyno::VirtualParticleGenerator<TDataType>;
	std::string pyclass_name = std::string("VirtualFissionFusionStrategy") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>DVFFS(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());

	py::enum_<typename Class::CandidatePointCount>(DVFFS, "CandidatePointCount")
		.value("neighbors_8", Class::CandidatePointCount::neighbors_8)
		.value("neighbors_27", Class::CandidatePointCount::neighbors_27)
		.value("neighbors_33", Class::CandidatePointCount::neighbors_33)
		.value("neighbors_125", Class::CandidatePointCount::neighbors_125)
		.export_values();

	py::enum_<typename Class::StretchedRegionCriteria>(DVFFS, "StretchedRegionCriteria")
		.value("Divergecne", Class::StretchedRegionCriteria::Divergecne)
		.value("ThinSheet", Class::StretchedRegionCriteria::ThinSheet)
		.value("Hybrid", Class::StretchedRegionCriteria::Hybrid)
		.export_values();

	DVFFS.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("fissionJudger", &Class::fissionJudger)
		.def("splitParticleArray", &Class::splitParticleArray)
		.def("constructFissionVirtualParticles", &Class::constructFissionVirtualParticles)
		.def("mergeVirtualParticles", &Class::mergeVirtualParticles)

		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inSmoothingLength", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("inSamplingDistance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("outDensity", &Class::outDensity, py::return_value_policy::reference)
		.def("inRPosition", &Class::inRPosition, py::return_value_policy::reference)
		.def("inRVelocity", &Class::inRVelocity, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("inThinSheet", &Class::inThinSheet, py::return_value_policy::reference)
		.def("inThinFeature", &Class::inThinFeature, py::return_value_policy::reference)
		.def("varTransitionRegionThreshold", &Class::varTransitionRegionThreshold, py::return_value_policy::reference)
		.def("inFrameNumber", &Class::inFrameNumber, py::return_value_policy::reference)
		.def("varMinDist", &Class::varMinDist, py::return_value_policy::reference)
		.def("outCandidateVirtualPoints", &Class::outCandidateVirtualPoints, py::return_value_policy::reference)
		.def("outVirtualPointType", &Class::outVirtualPointType, py::return_value_policy::reference)
		.def("vardeleteRepeatPoints", &Class::vardeleteRepeatPoints, py::return_value_policy::reference);
}

#include "DualParticleSystem/Module/VirtualSpatiallyAdaptiveStrategy.h"
template <typename TDataType>
void declare_virtual_spatially_adaptive_strategy(py::module& m, std::string typestr) {
	using Class = dyno::VirtualSpatiallyAdaptiveStrategy<TDataType>;
	using Parent = dyno::VirtualParticleGenerator<TDataType>;
	std::string pyclass_name = std::string("VirtualSpatiallyAdaptiveStrategy") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>VSAS(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());

	py::enum_<typename Class::CandidatePointCount>(VSAS, "CandidatePointCount")
		.value("neighbors_8", Class::CandidatePointCount::neighbors_8)
		.value("neighbors_27", Class::CandidatePointCount::neighbors_27)
		.value("neighbors_33", Class::CandidatePointCount::neighbors_33)
		.value("neighbors_125", Class::CandidatePointCount::neighbors_125)
		.export_values();

	VSAS.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("setHashGridSize", &Class::setHashGridSize)
		.def("varCandidatePointCount", &Class::varCandidatePointCount, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("varSamplingDistance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("inRPosition", &Class::inRPosition, py::return_value_policy::reference);
}


void pybind_dual_particle_system(py::module& m)
{
	declare_dual_particle_fluid_system<dyno::DataType3f>(m, "3f");
	declare_dual_particle_isph_module<dyno::DataType3f>(m, "3f");
	declare_energy_analyish<dyno::DataType3f>(m, "3f");
	declare_flip_fluid_explicit_solver<dyno::DataType3f>(m, "3f");
	declare_paticle_uniform_analysis<dyno::DataType3f>(m, "3f");
	declare_thin_feature<dyno::DataType3f>(m, "3f");
	declare_virtual_particle_generator<dyno::DataType3f>(m, "3f");
	declare_virtual_fission_fusion_strategy<dyno::DataType3f>(m, "3f");
	declare_virtual_colocation_strategy<dyno::DataType3f>(m, "3f");
	declare_virtual_particle_shifting_strategy<dyno::DataType3f>(m, "3f");
	declare_virtual_spatially_adaptive_strategy<dyno::DataType3f>(m, "3f");
}