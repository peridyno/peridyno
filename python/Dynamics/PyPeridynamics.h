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

	class LinearElasticitySolverPublicist : public Class
	{
	public:
		using Class::mBulkStiffness;
		using Class::mWeights;
		using Class::mDisplacement;
		using Class::mPosBuf;
		using Class::mF;
		using Class::mInvK;
	};

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
		.def("varIterationNumber", &Class::varIterationNumber, py::return_value_policy::reference)
		// protected
		.def_readwrite("mBulkStiffness", &LinearElasticitySolverPublicist::mBulkStiffness)
		.def_readwrite("mWeights", &LinearElasticitySolverPublicist::mWeights)
		.def_readwrite("mDisplacement", &LinearElasticitySolverPublicist::mDisplacement)
		.def_readwrite("mPosBuf", &LinearElasticitySolverPublicist::mPosBuf)
		.def_readwrite("mF", &LinearElasticitySolverPublicist::mF)
		.def_readwrite("mInvK", &LinearElasticitySolverPublicist::mInvK);
}

#include "Peridynamics/Module/CoSemiImplicitHyperelasticitySolver.h"
template <typename TDataType>
void declare_co_semi_implicit_hyperelasticity_solver(py::module& m, std::string typestr) {
	using Class = dyno::CoSemiImplicitHyperelasticitySolver<TDataType>;
	using Parent = dyno::LinearElasticitySolver<TDataType>;

	class CoSemiImplicitHyperelasticitySolverPublicist : public Class
	{
	public:
		using Class::initializeVolume;
		using Class::enforceHyperelasticity;
		using Class::resizeAllFields;
	};

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
		.def("setGrad_res_eps", &Class::setGrad_res_eps)
		.def("setAccelerated", &Class::setAccelerated)
		.def("getContactRulePtr", &Class::getContactRulePtr)
		// protected
		.def("initializeVolume", &CoSemiImplicitHyperelasticitySolverPublicist::initializeVolume)
		.def("enforceHyperelasticity", &CoSemiImplicitHyperelasticitySolverPublicist::enforceHyperelasticity)
		.def("resizeAllFields", &CoSemiImplicitHyperelasticitySolverPublicist::resizeAllFields);
}

#include "Peridynamics/Module/DragSurfaceInteraction.h"
template <typename TDataType>
void declare_drag_surface_interaction(py::module& m, std::string typestr) {
	using Class = dyno::DragSurfaceInteraction<TDataType>;
	using Parent = dyno::MouseInputModule;

	class DragSurfaceInteractionTrampoline : public Class
	{
	public:
		using Class::Class;

		void onEvent(dyno::PMouseEvent event) override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::DragSurfaceInteraction<TDataType>,
				onEvent,
				event
			);
		}

	};

	class DragSurfaceInteractionPublicist : public Class
	{
	public:
		using Class::onEvent;
	};

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
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)

		// protected
		.def("onEvent", &DragSurfaceInteractionPublicist::onEvent);
}

#include "Peridynamics/Module/DragVertexInteraction.h"
template <typename TDataType>
void declare_drag_vertex_interaction(py::module& m, std::string typestr) {
	using Class = dyno::DragVertexInteraction<TDataType>;
	using Parent = dyno::MouseInputModule;

	class DragVertexInteractionTrampoline : public Class
	{
	public:
		using Class::Class;

		void onEvent(dyno::PMouseEvent event) override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::DragVertexInteraction<TDataType>,
				onEvent,
				event
			);
		}

	};

	class DragVertexInteractionPublicist : public Class
	{
	public:
		using Class::onEvent;
	};

	std::string pyclass_name = std::string("DragVertexInteraction") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>()) 
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
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		// protected
		.def("onEvent", &DragVertexInteractionPublicist::onEvent);
}

#include "Peridynamics/Module/ElastoplasticityModule.h"
#include "cmath"
template <typename TDataType>
void declare_elastoplasticity_module(py::module& m, std::string typestr) {
	using Class = dyno::ElastoplasticityModule<TDataType>;
	using Parent = dyno::LinearElasticitySolver<TDataType>;

	class ElastoplasticityModuleTrampoline : public Class
	{
	public:
		using Class::Class;

		void constrain() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ElastoplasticityModule<TDataType>,
				constrain
			);
		}

		void solveElasticity() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ElastoplasticityModule<TDataType>,
				solveElasticity
			);
		}

		void applyPlasticity() override
		{
			PYBIND11_OVERRIDE_PURE(
				void,
				dyno::ElastoplasticityModule<TDataType>,
				applyPlasticity
			);
		}

	};

	class ElastoplasticityModulePublicist : public Class
	{
	public:
		using Class::constrain;
		using Class::solveElasticity;
		using Class::applyPlasticity;
		using Class::applyYielding;
		using Class::rotateRestShape;
		using Class::reconstructRestShape;
		using Class::computeA;
		using Class::computeB;
	};

	std::string pyclass_name = std::string("ElastoplasticityModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varCohesion", &Class::varCohesion, py::return_value_policy::reference)
		.def("varFrictionAngle", &Class::varFrictionAngle, py::return_value_policy::reference)
		.def("varIncompressible", &Class::varIncompressible, py::return_value_policy::reference)
		.def("varRenewNeighborhood", &Class::varRenewNeighborhood, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)

		// protected
		.def("constrain", &ElastoplasticityModulePublicist::constrain)
		.def("solveElasticity", &ElastoplasticityModulePublicist::solveElasticity)
		.def("applyPlasticity", &ElastoplasticityModulePublicist::applyPlasticity)
		.def("applyYielding", &ElastoplasticityModulePublicist::applyYielding)
		.def("rotateRestShape", &ElastoplasticityModulePublicist::rotateRestShape)
		.def("reconstructRestShape", &ElastoplasticityModulePublicist::reconstructRestShape)
		.def("computeA", &ElastoplasticityModulePublicist::computeA)
		.def("computeB", &ElastoplasticityModulePublicist::computeB);
}

#include "Peridynamics/Module/FixedPoints.h"
template <typename TDataType>
void declare_fixed_points(py::module& m, std::string typestr) {
	using Class = dyno::FixedPoints<TDataType>;
	using Parent = dyno::ConstraintModule;

	class FixedPointsTrampoline : public Class
	{
	public:
		using Class::Class;

		bool initializeImpl() override
		{
			PYBIND11_OVERRIDE_PURE(
				bool,
				dyno::FixedPoints<TDataType>,
				initializeImpl
			);
		}
	};

	class FixedPointsPublicist : public Class
	{
	public:
		using Class::initializeImpl;
		using Class::m_initPosID;
	};

	std::string pyclass_name = std::string("FixedPoints") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("addFixedPoint", &Class::addFixedPoint)
		.def("removeFixedPoint", &Class::removeFixedPoint)
		.def("clear", &Class::clear)

		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)

		.def_readwrite("FixedIds", &Class::FixedIds)
		.def_readwrite("FixedPos", &Class::FixedPos)


		// protected
		.def("initializeImpl", &FixedPointsPublicist::initializeImpl)
		.def_readwrite("m_initPosID", &FixedPointsPublicist::m_initPosID);
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

	class GranularModuleTrampoline : public Class
	{
	public:
		using Class::Class;

		void computeMaterialStiffness() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::GranularModule<TDataType>,
				computeMaterialStiffness
			);
		}
	};

	class GranularModulePublicist : public Class
	{
	public:
		using Class::computeMaterialStiffness;
	};

	std::string pyclass_name = std::string("GranularModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())

		// protected
		.def("computeMaterialStiffness", &GranularModulePublicist::computeMaterialStiffness);
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

	class SemiImplicitHyperelasticitySolverPublicist : public Class
	{
	public:
		using Class::enforceHyperelasticity;
		using Class::resizeAllFields;
	};

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

		.def("inVolumePair", &Class::inVolumePair, py::return_value_policy::reference)
		// protected
		.def("enforceHyperelasticity", &SemiImplicitHyperelasticitySolverPublicist::enforceHyperelasticity)
		.def("resizeAllFields", &SemiImplicitHyperelasticitySolverPublicist::resizeAllFields);
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
		.def_readwrite("mu", &Class::mu)
		.def_readwrite("xi", &Class::xi);
}

#include "Peridynamics/TriangularSystem.h"
template <typename TDataType>
void declare_triangular_system(py::module& m, std::string typestr) {
	using Class = dyno::TriangularSystem<TDataType>;
	using Parent = dyno::Node;

	class TriangularSystemTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TriangularSystem<TDataType>,
				resetStates
			);
		}

		void postUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TriangularSystem<TDataType>,
				postUpdateStates
			);
		}
	};

	class TriangularSystemPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::postUpdateStates;
	};

	std::string pyclass_name = std::string("TriangularSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)

		.def("stateTriangleSet", &Class::stateTriangleSet, py::return_value_policy::reference)
		.def("statePosition", &Class::statePosition, py::return_value_policy::reference)
		.def("stateVelocity", &Class::stateVelocity, py::return_value_policy::reference)

		.def("loadSurface", &Class::loadSurface)

		// protected
		.def("resetStates", &TriangularSystemPublicist::resetStates)
		.def("postUpdateStates", &TriangularSystemPublicist::postUpdateStates);
}

#include "Peridynamics/Cloth.h"
template <typename TDataType>
void declare_cloth(py::module& m, std::string typestr) {
	using Class = dyno::Cloth<TDataType>;
	using Parent = dyno::TriangularSystem<TDataType>;

	class ClothTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Cloth<TDataType>,
				resetStates
			);
		}

		void postUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Cloth<TDataType>,
				postUpdateStates
			);
		}
	};

	class ClothPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::postUpdateStates;
	};

	std::string pyclass_name = std::string("Cloth") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varHorizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("stateHorizon", &Class::stateHorizon, py::return_value_policy::reference)
		.def("state_rest_rotation", &Class::stateRestPosition, py::return_value_policy::reference)
		.def("stateOldPosition", &Class::stateOldPosition, py::return_value_policy::reference)
		.def("stateBonds", &Class::stateBonds, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ClothPublicist::resetStates)
		.def("postUpdateStates", &ClothPublicist::postUpdateStates);
}

#include "Peridynamics/CodimensionalPD.h"
template <typename TDataType>
void declare_codimensionalPD(py::module& m, std::string typestr) {
	using Class = dyno::CodimensionalPD<TDataType>;
	using Parent = dyno::TriangularSystem<TDataType>;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Real Real;

	class CodimensionalPDTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CodimensionalPD<TDataType>,
				resetStates
			);
		}

		void preUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CodimensionalPD<TDataType>,
				preUpdateStates
			);
		}

		void updateTopology() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::CodimensionalPD<TDataType>,
				updateTopology
			);
		}

		void updateRestShape() override
		{
			PYBIND11_OVERRIDE_PURE(
				void,
				dyno::CodimensionalPD<TDataType>,
				updateRestShape
			);
		}

		void updateVolume() override
		{
			PYBIND11_OVERRIDE_PURE(
				void,
				dyno::CodimensionalPD<TDataType>,
				updateVolume
			);
		}
	};

	class CodimensionalPDPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::preUpdateStates;
		using Class::updateTopology;
		using Class::updateRestShape;
		using Class::updateVolume;
	};


	std::string pyclass_name = std::string("CodimensionalPD") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		.def("scale", py::overload_cast<Real>(&Class::scale))
		.def("scale", py::overload_cast<Coord>(&Class::scale))
		.def("loadSurface", &Class::loadSurface)

		.def("setEnergyModel", py::overload_cast<dyno::StVKModel<Real>>(&Class::setEnergyModel),
			"Set energy model to StVKModel")
		.def("setEnergyModel", py::overload_cast<dyno::LinearModel<Real>>(&Class::setEnergyModel),
			"Set energy model to LinearModel")
		.def("setEnergyModel", py::overload_cast<dyno::NeoHookeanModel<Real>>(&Class::setEnergyModel),
			"Set energy model to NeoHookeanModel")
		.def("setEnergyModel", py::overload_cast<dyno::XuModel<Real>>(&Class::setEnergyModel),
			"Set energy model to XuModel")
		.def("setEnergyModel", py::overload_cast<dyno::FiberModel<Real>>(&Class::setEnergyModel),
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
		.def("stateMinLength", &Class::stateMinLength, py::return_value_policy::reference)
		// protected
		.def("resetStates", &CodimensionalPDPublicist::resetStates)
		.def("preUpdateStates", &CodimensionalPDPublicist::preUpdateStates)
		.def("updateTopology", &CodimensionalPDPublicist::updateTopology)
		.def("updateRestShape", &CodimensionalPDPublicist::updateRestShape)
		.def("updateVolume", &CodimensionalPDPublicist::updateVolume);
}

#include "Peridynamics/Peridynamics.h"
template <typename TDataType>
void declare_peridynamics(py::module& m, std::string typestr) {
	using Class = dyno::Peridynamics<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;

	class PeridynamicsTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Peridynamics<TDataType>,
				resetStates
			);
		}
	};

	class PeridynamicsPublicist : public Class
	{
	public:
		using Class::resetStates;
	};


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
		.def("stateBonds", &Class::stateBonds, py::return_value_policy::reference)
		// protected
		.def("resetStates", &PeridynamicsPublicist::resetStates);
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

	class HyperelasticBodyTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::HyperelasticBody<TDataType>,
				resetStates
			);
		}

	};

	class HyperelasticBodyPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateRestShape;
		using Class::updateVolume;
	};

	std::string pyclass_name = std::string("HyperelasticBody") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		.def("scale", py::overload_cast<Real>(&Class::scale))
		.def("scale", py::overload_cast<Coord>(&Class::scale))
		.def("rotate", py::overload_cast<dyno::Quat<Real>>(&Class::rotate))
		.def("rotate", py::overload_cast<Coord>(&Class::rotate))

		.def("setEnergyModel", py::overload_cast<dyno::StVKModel<Real>>(&Class::setEnergyModel),
			"Set energy model to StVKModel")
		.def("setEnergyModel", py::overload_cast<dyno::LinearModel<Real>>(&Class::setEnergyModel),
			"Set energy model to LinearModel")
		.def("setEnergyModel", py::overload_cast<dyno::NeoHookeanModel<Real>>(&Class::setEnergyModel),
			"Set energy model to NeoHookeanModel")
		.def("setEnergyModel", py::overload_cast<dyno::XuModel<Real>>(&Class::setEnergyModel),
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
		.def("stateTets", &Class::stateTets, py::return_value_policy::reference)
		// protected
		.def("resetStates", &HyperelasticBodyPublicist::resetStates)
		.def("updateRestShape", &HyperelasticBodyPublicist::updateRestShape)
		.def("updateVolume", &HyperelasticBodyPublicist::updateVolume);
}

#include "Peridynamics/TetrahedralSystem.h"
template <typename TDataType>
void declare_tetrahedral_system(py::module& m, std::string typestr) {
	using Class = dyno::TetrahedralSystem<TDataType>;
	using Parent = dyno::Node;

	class TetrahedralSystemTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TetrahedralSystem<TDataType>,
				resetStates
			);
		}

		void updateTopology() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TetrahedralSystem<TDataType>,
				updateTopology
			);
		}
	};

	class TetrahedralSystemPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateTopology;
	};

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
		.def("rotate", &Class::rotate)
		// protected
		.def("resetStates", &TetrahedralSystemPublicist::resetStates)
		.def("updateTopology", &TetrahedralSystemPublicist::updateTopology);
}

#include "Peridynamics/ThreadSystem.h"
template <typename TDataType>
void declare_thread_system(py::module& m, std::string typestr) {
	using Class = dyno::ThreadSystem<TDataType>;
	using Parent = dyno::Node;

	class ThreadSystemTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ThreadSystem<TDataType>,
				resetStates
			);
		}

		void updateTopology() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ThreadSystem<TDataType>,
				updateTopology
			);
		}
	};

	class ThreadSystemPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateTopology;
	};

	std::string pyclass_name = std::string("ThreadSystem") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str())
		.def(py::init<>())
		.def("stateEdgeSet", &Class::stateEdgeSet, py::return_value_policy::reference)
		.def("statePosition", &Class::statePosition, py::return_value_policy::reference)
		.def("stateVelocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("stateForce", &Class::stateForce, py::return_value_policy::reference)

		.def("addThread", &Class::addThread)
		// protected
		.def("resetStates", &ThreadSystemPublicist::resetStates)
		.def("updateTopology", &ThreadSystemPublicist::updateTopology);
}

#include "Peridynamics/Thread.h"
template <typename TDataType>
void declare_thread(py::module& m, std::string typestr) {
	using Class = dyno::Thread<TDataType>;
	using Parent = dyno::ThreadSystem<TDataType>;

	class ThreadTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::Thread<TDataType>,
				resetStates
			);
		}
	};

	class ThreadPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("Thread") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("translate", &Class::translate)
		.def("scale", &Class::scale)
		.def("varHorizon", &Class::varHorizon, py::return_value_policy::reference)
		.def("stateRestShape", &Class::stateRestShape, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ThreadPublicist::resetStates);
}

void pybind_peridynamics(py::module& m);