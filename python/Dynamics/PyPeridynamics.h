#pragma once
#include "../PyCommon.h"

#include "Peridynamics/Module/CalculateNormalSDF.h"
template <typename TDataType>
void declare_calculate_normal_sdf(py::module& m, std::string typestr) {
	using Class = dyno::CalculateNormalSDF<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("CalculateNormalSDF") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("compute", &Class::compute)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inNormalSDF", &Class::inNormalSDF, py::return_value_policy::reference)
		.def("inDisranceSDF", &Class::inDisranceSDF, py::return_value_policy::reference)
		.def("inTets", &Class::inTets, py::return_value_policy::reference);
}

#include "Peridynamics/Module/ContactRule.h"
template <typename TDataType>
void declare_contact_rule(py::module& m, std::string typestr) {
	using Class = dyno::ContactRule<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("ContactRule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("initCCDBroadPhase", &Class::initCCDBroadPhase)
		.def("setContactMaxIte", &Class::setContactMaxIte)

		.def("inTriangularMesh", &Class::inTriangularMesh, py::return_value_policy::reference)
		.def("varXi", &Class::varXi, py::return_value_policy::reference)
		.def("varS", &Class::varS, py::return_value_policy::reference)
		.def("inUnit", &Class::inUnit, py::return_value_policy::reference)
		.def("inOldPosition", &Class::inOldPosition, py::return_value_policy::reference)
		.def("inNewPosition", &Class::inNewPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)

		.def("outContactForce", &Class::outContactForce, py::return_value_policy::reference)
		.def("outWeight", &Class::outWeight, py::return_value_policy::reference)
		.def_readwrite("weight", &Class::weight);
}

#include "Peridynamics/Module/LinearElasticitySolver.h"
template <typename TDataType>
void declare_linear_elasticity_solver(py::module& m, std::string typestr) {
	using Class = dyno::LinearElasticitySolver<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("LinearElasticitySolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inHorizon", &Class::inHorizon, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inX", &Class::inX, py::return_value_policy::reference)
		.def("inY", &Class::inY, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inBonds", &Class::inBonds, py::return_value_policy::reference)
		.def("varMu", &Class::varMu, py::return_value_policy::reference)
		.def("varLambda", &Class::varLambda, py::return_value_policy::reference)
		.def("varIterationNumber", &Class::varIterationNumber, py::return_value_policy::reference);
}

#include "Peridynamics/Module/CoSemiImplicitHyperelasticitySolver.h"
template <typename TDataType>
void declare_co_semi_implicit_hyperelasticity_solver(py::module& m, std::string typestr) {
	using Class = dyno::CoSemiImplicitHyperelasticitySolver<TDataType>;
	using Parent = dyno::LinearElasticitySolver<TDataType>;
	std::string pyclass_name = std::string("CoSemiImplicitHyperelasticitySolverr") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("solveElasticity", &Class::solveElasticity)
		.def("setObjectVolume", &Class::setObjectVolume)
		.def("setParticleVolume", &Class::setParticleVolume)
		.def("setContactMaxIte", &Class::setContactMaxIte)
		.def("inEnergyType", &Class::inEnergyType, py::return_value_policy::reference)
		.def("inEnergyModels", &Class::inEnergyModels, py::return_value_policy::reference)
		.def("varNeighborSearchingAdjacent", &Class::varNeighborSearchingAdjacent, py::return_value_policy::reference)
		.def("inRestNorm", &Class::inRestNorm, py::return_value_policy::reference)
		.def("inNorm", &Class::inNorm, py::return_value_policy::reference)
		.def("inOldPosition", &Class::inOldPosition, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inUnit", &Class::inUnit, py::return_value_policy::reference)
		.def("inTriangularMesh", &Class::inTriangularMesh, py::return_value_policy::reference)
		.def("setXi", &Class::setXi)
		.def("setK_bend", &Class::setK_bend)
		.def("setSelfContact", &Class::setSelfContact)
		.def("getXi", &Class::getXi)
		.def("setE", &Class::setE)
		.def("getE", &Class::getE)
		.def("setS", &Class::setS)
		.def("getS", &Class::getS)
		.def("getS0", &Class::getS0)
		.def("getS1", &Class::getS1)
		.def("get_grad_res_eps", &Class::setGrad_res_eps)
		.def("setAccelerated", &Class::setAccelerated)
		.def("getContactRulePtr", &Class::getContactRulePtr);
}

#include "Peridynamics/Module/DragSurfaceInteraction.h"
template <typename TDataType>
void declare_drag_surface_interaction(py::module& m, std::string typestr) {
	using Class = dyno::DragSurfaceInteraction<TDataType>;
	using Parent = dyno::MouseInputModule;
	std::string pyclass_name = std::string("DragSurfaceInteraction") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>()) // °ó¶¨Ä¬ÈÏ¹¹Ôìº¯Êý
		.def("InteractionClick", &Class::InteractionClick)
		.def("InteractionDrag", &Class::InteractionDrag)
		.def("calcSurfaceInteractClick", &Class::calcSurfaceInteractClick)
		.def("calcSurfaceInteractDrag", &Class::calcSurfaceInteractDrag)
		.def("setTriFixed", &Class::setTriFixed)
		.def("cancelVelocity", &Class::cancelVelocity)

		.def("inInitialTriangleSet", &Class::inInitialTriangleSet, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("varInterationRadius", &Class::varInterationRadius, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference);
}

#include "Peridynamics/Module/DragVertexInteraction.h"
template <typename TDataType>
void declare_drag_vertex_interaction(py::module& m, std::string typestr) {
	using Class = dyno::DragVertexInteraction<TDataType>;
	using Parent = dyno::MouseInputModule;
	std::string pyclass_name = std::string("DragVertexInteraction") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>()) // °ó¶¨Ä¬ÈÏ¹¹Ôìº¯Êý
		.def("InteractionClick", &Class::InteractionClick)
		.def("InteractionDrag", &Class::InteractionDrag)
		.def("calcVertexInteractClick", &Class::calcVertexInteractClick)
		.def("calcVertexInteractDrag", &Class::calcVertexInteractDrag)
		.def("setVertexFixed", &Class::setVertexFixed)
		.def("cancelVelocity", &Class::cancelVelocity)

		.def("inInitialTriangleSet", &Class::inInitialTriangleSet, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("varInterationRadius", &Class::varInterationRadius, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference);
}

#include "Peridynamics/Module/ElastoplasticityModule.h"
#include "cmath"
template <typename TDataType>
void declare_elastoplasticity_module(py::module& m, std::string typestr) {
	using Class = dyno::ElastoplasticityModule<TDataType>;
	using Parent = dyno::LinearElasticitySolver<TDataType>;
	std::string pyclass_name = std::string("ElastoplasticityModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varCohesion", &Class::varCohesion, py::return_value_policy::reference)
		.def("varFrictionAngle", &Class::varFrictionAngle, py::return_value_policy::reference)
		.def("varIncompressible", &Class::varIncompressible, py::return_value_policy::reference)
		.def("varRenewNeighborhood", &Class::varRenewNeighborhood, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference);
}

#include "Peridynamics/Module/FixedPoints.h"
template <typename TDataType>
void declare_fixed_points(py::module& m, std::string typestr) {
	using Class = dyno::FixedPoints<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("FixedPoints") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("addFixedPoint", &Class::addFixedPoint)
		.def("removeFixedPoint", &Class::removeFixedPoint)
		.def("clear", &Class::clear)

		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)

		.def_readwrite("FixedIds", &Class::FixedIds)
		.def_readwrite("FixedPos", &Class::FixedPos);
}

#include "Peridynamics/Module/FractureModule.h"
template <typename TDataType>
void declare_fracture_module(py::module& m, std::string typestr) {
	using Class = dyno::FractureModule<TDataType>;
	using Parent = dyno::ElastoplasticityModule<TDataType>;
	std::string pyclass_name = std::string("FractureModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("applyPlasticity", &Class::applyPlasticity);
}

#include "Peridynamics/Module/GranularModule.h"
template <typename TDataType>
void declare_granular_module(py::module& m, std::string typestr) {
	using Class = dyno::GranularModule<TDataType>;
	using Parent = dyno::ElastoplasticityModule<TDataType>;
	std::string pyclass_name = std::string("GranularModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

#include "Peridynamics/Module/ProjectivePeridynamics.h"
template <typename TDataType>
void declare_projective_peridynamics(py::module& m, std::string typestr) {
	using Class = dyno::ProjectivePeridynamics<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("ProjectivePeridynamics") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inHorizon", &Class::inHorizon, py::return_value_policy::reference)

		.def("inX", &Class::inX, py::return_value_policy::reference)
		.def("inY", &Class::inY, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)

		.def("inBonds", &Class::inBonds, py::return_value_policy::reference);
}

#include "Peridynamics/Module/SemiImplicitHyperelasticitySolver.h"
template <typename TDataType>
void declare_semi_implicit_hyperelasticity_solver(py::module& m, std::string typestr) {
	using Class = dyno::SemiImplicitHyperelasticitySolver<TDataType>;
	using Parent = dyno::LinearElasticitySolver<TDataType>;
	std::string pyclass_name = std::string("SemiImplicitHyperelasticitySolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("solveElasticity", &Class::solveElasticity)
		.def("varStrainLimiting", &Class::varStrainLimiting, py::return_value_policy::reference)

		.def("inEnergyType", &Class::inEnergyType, py::return_value_policy::reference)
		.def("inEnergyModels", &Class::inEnergyModels, py::return_value_policy::reference)
		.def("varIsAlphaComputed", &Class::varIsAlphaComputed, py::return_value_policy::reference)

		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inVolume", &Class::inVolume, py::return_value_policy::reference)

		.def("inVolumePair", &Class::inVolumePair, py::return_value_policy::reference);
}

#include "Peridynamics/Bond.h"
template <typename TDataType>
void declare_bond(py::module& m, std::string typestr) {
	using Class = dyno::TBond<TDataType>;
	std::string pyclass_name = std::string("TBond") + typestr;
	typedef typename TDataType::Coord Coord;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def(py::init<int, Coord>())
		.def_readwrite("idx", &Class::idx)
		.def_readwrite("xi", &Class::xi);
}

#include "Peridynamics/TriangularSystem.h"
template <typename TDataType>
void declare_triangular_system(py::module& m, std::string typestr) {
	using Class = dyno::TriangularSystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("TriangularSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)

		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("statePosition", &Class::statePosition, py::return_value_policy::reference)
		.def("stateVelocity", &Class::stateVelocity, py::return_value_policy::reference)

		.def("loadSurface", &Class::loadSurface);
}

#include "Peridynamics/Cloth.h"
template <typename TDataType>
void declare_cloth(py::module& m, std::string typestr) {
	using Class = dyno::Cloth<TDataType>;
	using Parent = dyno::TriangularSystem<TDataType>;
	std::string pyclass_name = std::string("Cloth") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varHorizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("stateHorizon", &Class::stateHorizon, py::return_value_policy::reference)
		.def("state_rest_rotation", &Class::stateRestPosition, py::return_value_policy::reference)
		.def("stateOldPosition", &Class::stateOldPosition, py::return_value_policy::reference)
		.def("stateBonds", &Class::stateBonds, py::return_value_policy::reference);
}

#include "Peridynamics/CodimensionalPD.h"
template <typename TDataType>
void declare_codimensionalPD(py::module& m, std::string typestr) {
	using Class = dyno::CodimensionalPD<TDataType>;
	using Parent = dyno::TriangularSystem<TDataType>;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Real Real;
	std::string pyclass_name = std::string("CodimensionalPD") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		.def("scale", py::overload_cast<Real>(&Class::scale))
		.def("scale", py::overload_cast<Coord>(&Class::scale))
		.def("loadSurface", &Class::loadSurface)

		.def("set_energy_model", py::overload_cast<dyno::StVKModel<Real>>(&Class::setEnergyModel),
			"Set energy model to StVKModel")
		.def("set_energy_model", py::overload_cast<dyno::LinearModel<Real>>(&Class::setEnergyModel),
			"Set energy model to LinearModel")
		.def("set_energy_model", py::overload_cast<dyno::NeoHookeanModel<Real>>(&Class::setEnergyModel),
			"Set energy model to NeoHookeanModel")
		.def("set_energy_model", py::overload_cast<dyno::XuModel<Real>>(&Class::setEnergyModel),
			"Set energy model to XuModel")
		.def("set_energy_model", py::overload_cast<dyno::FiberModel<Real>>(&Class::setEnergyModel),
			"Set energy model to FiberModel")
		//DEF_VAR
		.def("varHorizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("varEnergyType", &Class::varEnergyType, py::return_value_policy::reference)
		.def("varEnergyModel", &Class::varEnergyModel, py::return_value_policy::reference)
		//DEF_ARRAY_STATE
		.def("stateRestPosition", &Class::stateRestPosition, py::return_value_policy::reference)
		.def("stateOldPosition", &Class::stateOldPosition, py::return_value_policy::reference)
		.def("stateRestNorm", &Class::stateRestNorm, py::return_value_policy::reference)
		.def("stateNorm", &Class::stateNorm, py::return_value_policy::reference)
		.def("stateAttribute", &Class::stateAttribute, py::return_value_policy::reference)
		.def("stateVolume", &Class::stateVolume, py::return_value_policy::reference)
		//DEF_ARRAYLIST_STATE
		.def("stateRestShape", &Class::stateRestShape, py::return_value_policy::reference)
		//DEF_VAR_STATE
		.def("stateMaxLength", &Class::stateMaxLength, py::return_value_policy::reference)
		.def("stateMinLength", &Class::stateMinLength, py::return_value_policy::reference);
}

#include "Peridynamics/Peridynamics.h"
template <typename TDataType>
void declare_peridynamics(py::module& m, std::string typestr) {
	using Class = dyno::Peridynamics<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;
	std::string pyclass_name = std::string("Peridynamics") + typestr;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename dyno::TBond<TDataType> Bond;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)

		.def("importSolidParticles", &Class::importSolidParticles, py::return_value_policy::reference)
		.def("getSolidParticles", &Class::getSolidParticles)
		.def("addSolidParticle", &Class::addSolidParticle)
		.def("removeSolidParticle", &Class::removeSolidParticle)

		.def("varHorizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("stateHorizon", &Class::stateHorizon, py::return_value_policy::reference)
		.def("stateReferencePosition", &Class::stateReferencePosition, py::return_value_policy::reference)
		.def("stateBonds", &Class::stateBonds, py::return_value_policy::reference);
}

#include "Peridynamics/ElasticBody.h"
template <typename TDataType>
void declare_elastic_body(py::module& m, std::string typestr) {
	using Class = dyno::ElasticBody<TDataType>;
	using Parent = dyno::Peridynamics<TDataType>;
	std::string pyclass_name = std::string("ElasticBody") + typestr;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Real Real;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol())
		.def(py::init<>());
}

#include "Peridynamics/ElastoplasticBody.h"
template <typename TDataType>
void declare_elastoplastic_body(py::module& m, std::string typestr) {
	using Class = dyno::ElastoplasticBody<TDataType>;
	using Parent = dyno::Peridynamics<TDataType>;
	std::string pyclass_name = std::string("ElastoplasticBody") + typestr;
	typedef typename TDataType::Coord Coord;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

#include "Peridynamics/HyperelasticBody.h"
template <typename TDataType>
void declare_hyperelastic_body(py::module& m, std::string typestr) {
	using Class = dyno::HyperelasticBody<TDataType>;
	using Parent = dyno::TetrahedralSystem<TDataType>;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	std::string pyclass_name = std::string("HyperelasticBody") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		.def("scale", py::overload_cast<Real>(&Class::scale))
		.def("scale", py::overload_cast<Coord>(&Class::scale))
		.def("rotate", py::overload_cast<dyno::Quat<Real>>(&Class::rotate))
		.def("rotate", py::overload_cast<Coord>(&Class::rotate))

		.def("set_energy_model", py::overload_cast<dyno::StVKModel<Real>>(&Class::setEnergyModel),
			"Set energy model to StVKModel")
		.def("set_energy_model", py::overload_cast<dyno::LinearModel<Real>>(&Class::setEnergyModel),
			"Set energy model to LinearModel")
		.def("set_energy_model", py::overload_cast<dyno::NeoHookeanModel<Real>>(&Class::setEnergyModel),
			"Set energy model to NeoHookeanModel")
		.def("set_energy_model", py::overload_cast<dyno::XuModel<Real>>(&Class::setEnergyModel),
			"Set energy model to XuModel")

		.def("loadSDF", &Class::loadSDF)

		.def("varLocation", &Class::varLocation, py::return_value_policy::reference)
		.def("varRotation", &Class::varRotation, py::return_value_policy::reference)
		.def("varScale", &Class::varScale, py::return_value_policy::reference)
		.def("varHorizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("varAlphaComputed", &Class::varAlphaComputed, py::return_value_policy::reference)
		.def("varEnergyType", &Class::varEnergyType, py::return_value_policy::reference)
		.def("varEnergyModel", &Class::varEnergyModel, py::return_value_policy::reference)
		.def("stateRestPosition", &Class::stateRestPosition, py::return_value_policy::reference)
		.def("stateBonds", &Class::stateBonds, py::return_value_policy::reference)
		.def("stateVolumePair", &Class::stateVolumePair, py::return_value_policy::reference)

		.def("stateVertexRotation", &Class::stateVertexRotation, py::return_value_policy::reference)
		.def("stateAttribute", &Class::stateAttribute, py::return_value_policy::reference)
		.def("stateVolume", &Class::stateVolume, py::return_value_policy::reference)
		.def("varNeighborSearchingAdjacent", &Class::varNeighborSearchingAdjacent, py::return_value_policy::reference)
		.def("varFileName", &Class::varFileName, py::return_value_policy::reference)
		.def("stateTets", &Class::stateTets, py::return_value_policy::reference);
}

#include "Peridynamics/TetrahedralSystem.h"
template <typename TDataType>
void declare_tetrahedral_system(py::module& m, std::string typestr) {
	using Class = dyno::TetrahedralSystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("TetrahedralSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("stateNormalSDF", &Class::stateNormalSDF, py::return_value_policy::reference)
		.def("varSDF", &Class::varSDF, py::return_value_policy::reference)
		.def("stateTetrahedronSet", &Class::stateTetrahedronSet, py::return_value_policy::reference)
		.def("statePosition", &Class::statePosition, py::return_value_policy::reference)
		.def("stateVelocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("stateForce", &Class::stateForce, py::return_value_policy::reference)

		.def("loadVertexFromFile", &Class::loadVertexFromFile)
		.def("loadVertexFromGmshFile", &Class::loadVertexFromGmshFile)
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)
		.def("rotate", &Class::rotate);
}

#include "Peridynamics/ThreadSystem.h"
template <typename TDataType>
void declare_thread_system(py::module& m, std::string typestr) {
	using Class = dyno::ThreadSystem<TDataType>;
	using Parent = dyno::Node;
	std::string pyclass_name = std::string("ThreadSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str())
		.def(py::init<>())
		.def("stateEdgeSet", &Class::stateEdgeSet, py::return_value_policy::reference)
		.def("statePosition", &Class::statePosition, py::return_value_policy::reference)
		.def("stateVelocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("stateForce", &Class::stateForce, py::return_value_policy::reference)

		.def("addThread", &Class::addThread);
}

#include "Peridynamics/Thread.h"
template <typename TDataType>
void declare_thread(py::module& m, std::string typestr) {
	using Class = dyno::Thread<TDataType>;
	using Parent = dyno::ThreadSystem<TDataType>;
	std::string pyclass_name = std::string("Thread") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)
		.def("varHorizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("stateRestShape", &Class::stateRestShape, py::return_value_policy::reference);
}

void pybind_peridynamics(py::module& m);