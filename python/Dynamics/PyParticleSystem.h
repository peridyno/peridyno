#pragma once
#include "../PyCommon.h"

#include "Array/Array.h"

#include "ParticleSystem/ParticleFluid.h"
#include "ParticleSystem/GhostParticles.h"

#include "Peridynamics/ElasticBody.h"
#include "Peridynamics/Module/LinearElasticitySolver.h"

#include "Peridynamics/TriangularSystem.h"

#include "RigidBody/RigidBody.h"

using Node = dyno::Node;
using NodePort = dyno::NodePort;

#include "ParticleSystem/Emitters/ParticleEmitter.h"
template <typename TDataType>
void declare_particle_emitter(py::module& m, std::string typestr) {
	using Class = dyno::ParticleEmitter<TDataType>;
	using Parent = dyno::ParametricModel<TDataType>;

	class ParticleEmitterTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ParticleEmitter<TDataType>,
				resetStates
			);
		}

		void generateParticles() override
		{
			PYBIND11_OVERRIDE_PURE(
				void,
				dyno::ParticleEmitter<TDataType>,
				generateParticles
			);
		}
	};

	class ParticleEmitterPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::updateStates;
		using Class::generateParticles;
		using Class::rotationMatrix;

		using Class::mPosition;
		using Class::mVelocity;
		using Class::mTimeInterval;
	};

	std::string pyclass_name = std::string("ParticleEmitter") + typestr;
	py::class_<Class, Parent, ParticleEmitterTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("sizeOfParticles", &Class::sizeOfParticles)

		.def("getPositions", &Class::getPositions)
		.def("getVelocities", &Class::getVelocities)

		.def("getNodeType", &Class::getNodeType)

		.def("varVelocityMagnitude", &Class::varVelocityMagnitude, py::return_value_policy::reference)
		.def("varSamplingDistance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("varSpacing", &Class::varSpacing, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ParticleEmitterPublicist::resetStates, py::return_value_policy::reference)
		.def("updateStates", &ParticleEmitterPublicist::updateStates, py::return_value_policy::reference)
		.def("generateParticles", &ParticleEmitterPublicist::generateParticles, py::return_value_policy::reference)
		.def("rotationMatrix", &ParticleEmitterPublicist::rotationMatrix, py::return_value_policy::reference)

		.def_readwrite("mPosition", &ParticleEmitterPublicist::mPosition)
		.def_readwrite("mVelocity", &ParticleEmitterPublicist::mVelocity)
		.def_readwrite("mTimeInterval", &ParticleEmitterPublicist::mTimeInterval);
}

#include "ParticleSystem/Emitters/CircularEmitter.h"
template <typename TDataType>
void declare_circular_emitter(py::module& m, std::string typestr) {
	using Class = dyno::CircularEmitter<TDataType>;
	using Parent = dyno::ParticleEmitter<TDataType>;
	std::string pyclass_name = std::string("CircularEmitter") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varRadius", &Class::varRadius, py::return_value_policy::reference)
		.def("stateOutline", &Class::stateOutline, py::return_value_policy::reference);
}

#include "ParticleSystem/Emitters/PoissonEmitter.h"
template <typename TDataType>
void declare_poisson_emitter(py::module& m, std::string typestr) {
	using Class = dyno::PoissonEmitter<TDataType>;
	using Parent = dyno::ParticleEmitter<TDataType>;

	class PoissonEmitterTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PoissonEmitter<TDataType>,
				resetStates
			);
		}

		void generateParticles() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::PoissonEmitter<TDataType>,
				generateParticles
			);
		}
	};

	class PoissonEmitterPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::generateParticles;
	};

	std::string pyclass_name = std::string("PoissonEmitter") + typestr;
	py::class_<Class, Parent, PoissonEmitterTrampoline, std::shared_ptr<Class>>PE(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	PE.def(py::init<>())
		.def("varWidth", &Class::varWidth, py::return_value_policy::reference)
		.def("varHeight", &Class::varHeight, py::return_value_policy::reference)
		.def("varDelayStart", &Class::varDelayStart, py::return_value_policy::reference)
		.def("stateOutline", &Class::stateOutline, py::return_value_policy::reference)
		.def("varEmitterShape", &Class::varEmitterShape, py::return_value_policy::reference)
		// protected
		.def("resetStates", &PoissonEmitterPublicist::resetStates, py::return_value_policy::reference)
		.def("generateParticles", &PoissonEmitterPublicist::generateParticles, py::return_value_policy::reference);

	py::enum_<typename Class::EmitterShape>(PE, "EmitterShape")
		.value("Square", Class::EmitterShape::Square)
		.value("Round", Class::EmitterShape::Round)
		.export_values();
}

#include "ParticleSystem/Emitters/SquareEmitter.h"
template <typename TDataType>
void declare_square_emitter(py::module& m, std::string typestr) {
	using Class = dyno::SquareEmitter<TDataType>;
	using Parent = dyno::ParticleEmitter<TDataType>;

	class SquareEmitterTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SquareEmitter<TDataType>,
				resetStates
			);
		}

		void generateParticles() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SquareEmitter<TDataType>,
				generateParticles
			);
		}
	};

	class SquareEmitterPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::generateParticles;
	};

	std::string pyclass_name = std::string("SquareEmitter") + typestr;
	py::class_<Class, Parent, SquareEmitterTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varWidth", &Class::varWidth, py::return_value_policy::reference)
		.def("varHeight", &Class::varHeight, py::return_value_policy::reference)
		.def("stateOutline", &Class::stateOutline, py::return_value_policy::reference)
		// protected
		.def("resetStates", &SquareEmitterPublicist::resetStates, py::return_value_policy::reference)
		.def("generateParticles", &SquareEmitterPublicist::generateParticles, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/ApproximateImplicitViscosity.h"
template <typename TDataType>
void declare_approximate_implicit_viscosity(py::module& m, std::string typestr) {
	using Class = dyno::ApproximateImplicitViscosity<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("ApproximateImplicitViscosity") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>AIV(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	AIV.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("SetCross", &Class::SetCross)

		.def("varViscosity", &Class::varViscosity, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("varLowerBoundViscosity", &Class::varLowerBoundViscosity, py::return_value_policy::reference)
		.def("varCrossK", &Class::varCrossK, py::return_value_policy::reference)
		.def("varCrossN", &Class::varCrossN, py::return_value_policy::reference)
		.def("varFluidType", &Class::varFluidType, py::return_value_policy::reference);

	py::enum_<typename Class::FluidType>(AIV, "FluidType")
		.value("NewtonianFluid", Class::FluidType::NewtonianFluid)
		.value("NonNewtonianFluid", Class::FluidType::NonNewtonianFluid)
		.export_values();
}

#include "ParticleSystem/Module/BoundaryConstraint.h"
template <typename TDataType>
void declare_boundary_constraint(py::module& m, std::string typestr) {
	using Class = dyno::BoundaryConstraint<TDataType>;
	using Parent = dyno::ConstraintModule;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	std::string pyclass_name = std::string("BoundaryConstraint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def("constrain", py::overload_cast<void>(&Class::constrain))
		.def("constrain", py::overload_cast<dyno::Array<Coord, DeviceType::GPU>&, dyno::Array<Coord, DeviceType::GPU>&, Real>(&Class::constrain))
		//.def("constrain", py::overload_cast<dyno::Array<Coord, DeviceType::GPU>&, dyno::Array<Coord, DeviceType::GPU>&, dyno::DistanceField3D<TDataType>&, Real>(&Class::constrain))
		.def("load", &Class::load)
		.def("setCube", &Class::setCube)
		.def("setSphere", &Class::setSphere)
		.def("setCylinder", &Class::setCylinder)
		.def("varTangentialFriction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("varNormalFriction", &Class::varNormalFriction, py::return_value_policy::reference)

		.def_readwrite("m_position", &Class::m_position)
		.def_readwrite("m_velocity", &Class::m_velocity);
}

#include "ParticleSystem/Module/ParticleApproximation.h"
template <typename TDataType>
void declare_particle_approximation(py::module& m, std::string typestr) {
	using Class = dyno::ParticleApproximation<TDataType>;
	using Parent = dyno::ComputeModule;

	class ParticleApproximationPublicist : public Class
	{
	public:
		using Class::mScalingFactor;
	};

	std::string pyclass_name = std::string("ParticleApproximation") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>PA(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
	PA.def(py::init<>())
		.def("compute", &Class::compute)
		.def("inSmoothingLength", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("inSamplingDistance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("varKernelType", &Class::varKernelType, py::return_value_policy::reference)

		.def_readwrite("mScalingFactor", &ParticleApproximationPublicist::mScalingFactor);

	py::enum_<typename Class::EKernelType>(PA, "EKernelType")
		.value("KT_Smooth", Class::EKernelType::KT_Smooth)
		.value("KT_Spiky", Class::EKernelType::KT_Spiky)
		.value("KT_Cubic", Class::EKernelType::KT_Cubic)
		.value("KT_Constant", Class::EKernelType::KT_Constant)
		.value("KT_Quartic", Class::EKernelType::KT_Quartic)
		.value("KT_Corrected", Class::EKernelType::KT_Corrected)
		.value("KT_CorrectedQuatic", Class::EKernelType::KT_CorrectedQuatic)
		.value("KT_WendlandC2", Class::EKernelType::KT_WendlandC2)
		.value("KT_CorrectedMPSKernel", Class::EKernelType::KT_CorrectedMPSKernel)
		.export_values();
}

#include "ParticleSystem/Module/DivergenceFreeSphSolver.h"
template <typename TDataType>
void declare_divergence_free_sph_solver(py::module& m, std::string typestr) {
	using Class = dyno::DivergenceFreeSphSolver<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("DivergenceFreeSphSolver") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("outDensity", &Class::outDensity, py::return_value_policy::reference)
		.def("varDivergenceSolverDisabled", &Class::varDivergenceSolverDisabled, py::return_value_policy::reference)
		.def("varDensitySolverDisabled", &Class::varDensitySolverDisabled, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("varDivergenceErrorThreshold", &Class::varDivergenceErrorThreshold, py::return_value_policy::reference)
		.def("varDensityErrorThreshold", &Class::varDensityErrorThreshold, py::return_value_policy::reference)
		.def("varMaxIterationNumber", &Class::varMaxIterationNumber, py::return_value_policy::reference)
		.def("compute", &Class::compute)
		.def("computeAlpha", &Class::computeAlpha)
		.def("takeOneDensityIteration", &Class::takeOneDensityIteration)
		.def("takeOneDivergenIteration", &Class::takeOneDivergenIteration);
}

#include "ParticleSystem/Module/ImplicitISPH.h"
template <typename TDataType>
void declare_implicit_ISPH(py::module& m, std::string typestr) {
	using Class = dyno::ImplicitISPH<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;

	class ImplicitISPHTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ImplicitISPH<TDataType>,
				compute
			);
		}
	};

	class ImplicitISPHPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("ImplicitISPH") + typestr;
	py::class_<Class, Parent, ImplicitISPHTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("outDensity", &Class::outDensity, py::return_value_policy::reference)
		.def("varIterationNumber", &Class::varIterationNumber, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("varKappa", &Class::varKappa, py::return_value_policy::reference)
		.def("varRelaxedOmega", &Class::varRelaxedOmega, py::return_value_policy::reference)
		.def("takeOneIteration", &Class::takeOneIteration)
		.def("PreIterationCompute", &Class::PreIterationCompute)
		.def("updateVelocity", &Class::updateVelocity)
		// protected
		.def("compute", &ImplicitISPHPublicist::compute, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/ImplicitViscosity.h"
template <typename TDataType>
void declare_implicit_viscosity(py::module& m, std::string typestr) {
	using Class = dyno::ImplicitViscosity<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("ImplicitViscosity") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varViscosity", &Class::varViscosity, py::return_value_policy::reference)
		.def("varInterationNumber", &Class::varInterationNumber, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("compute", &Class::compute);
}

#include "ParticleSystem/Module/IterativeDensitySolver.h"
template <typename TDataType>
void declare_iterative_densitySolver(py::module& m, std::string typestr) {
	using Class = dyno::IterativeDensitySolver<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;

	class IterativeDensitySolverTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::IterativeDensitySolver<TDataType>,
				compute
			);
		}
	};

	class IterativeDensitySolverPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("IterativeDensitySolver") + typestr;
	py::class_<Class, Parent, IterativeDensitySolverTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("outDensity", &Class::outDensity, py::return_value_policy::reference)
		.def("varIterationNumber", &Class::varIterationNumber, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("varKappa", &Class::varKappa, py::return_value_policy::reference)
		.def("takeOneIteration", &Class::takeOneIteration)
		.def("updateVelocity", &Class::updateVelocity)
		// protected
		.def("compute", &IterativeDensitySolverPublicist::compute, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/LinearDamping.h"
template <typename TDataType>
void declare_linear_damping(py::module& m, std::string typestr) {
	using Class = dyno::LinearDamping<TDataType>;
	using Parent = dyno::ConstraintModule;

	class LinearDampingTrampoline : public Class
	{
	public:
		using Class::Class;

		void constrain() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::LinearDamping<TDataType>,
				constrain
			);
		}
	};

	class LinearDampingPublicist : public Class
	{
	public:
		using Class::constrain;
	};


	std::string pyclass_name = std::string("LinearDamping") + typestr;
	py::class_<Class, Parent, LinearDampingTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varDampingCoefficient", &Class::varDampingCoefficient, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		// protected
		.def("constrain", &LinearDampingPublicist::constrain, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/NormalForce.h"
template <typename TDataType>
void declare_normal_force(py::module& m, std::string typestr) {
	using Class = dyno::NormalForce<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("NormalForce") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("outNormalForce", &Class::outNormalForce, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inParticleNormal", &Class::inParticleNormal, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("varStrength", &Class::varStrength, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inTriangleNeighborIds", &Class::inTriangleNeighborIds, py::return_value_policy::reference)
		.def("inParticleMeshID", &Class::inParticleMeshID, py::return_value_policy::reference)
		.def("constrain", &Class::constrain);
}

#include "ParticleSystem/Module/ParticleIntegrator.h"
template <typename TDataType>
void declare_particle_integrator(py::module& m, std::string typestr) {
	using Class = dyno::ParticleIntegrator<TDataType>;
	using Parent = dyno::ComputeModule;

	class ParticleIntegratorTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ParticleIntegrator<TDataType>,
				compute
			);
		}
	};

	class ParticleIntegratorPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("ParticleIntegrator") + typestr;
	py::class_<Class, Parent, ParticleIntegratorTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		// protected
		.def("compute", &ParticleIntegratorPublicist::compute, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/PositionBasedFluidModel.h"
template <typename TDataType>
void declare_position_based_fluid_model(py::module& m, std::string typestr) {
	using Class = dyno::PositionBasedFluidModel<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("PositionBasedFluidModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inSamplingDistance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("inSmoothingLength", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/ProjectionBasedFluidModel.h"
template <typename TDataType>
void declare_projection_based_fluid_model(py::module& m, std::string typestr) {
	using Class = dyno::ProjectionBasedFluidModel<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("ProjectionBasedFluidModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inSamplingDistance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("inSmoothingLength", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inNormal", &Class::inNormal, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/SemiImplicitDensitySolver.h"
template <typename TDataType>
void declare_SemiImplicitDensitySolver(py::module& m, std::string typestr) {
	using Class = dyno::SemiImplicitDensitySolver<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;

	class SemiImplicitDensitySolverTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SemiImplicitDensitySolver<TDataType>,
				compute
			);
		}
	};

	class SemiImplicitDensitySolverPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("SemiImplicitDensitySolver") + typestr;
	py::class_<Class, Parent, SemiImplicitDensitySolverTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("outDensity", &Class::outDensity, py::return_value_policy::reference)
		.def("varIterationNumber", &Class::varIterationNumber, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("varKappa", &Class::varKappa, py::return_value_policy::reference)
		.def("updatePosition", &Class::updatePosition)
		.def("updateVelocity", &Class::updateVelocity)
		// protected
		.def("compute", &SemiImplicitDensitySolverPublicist::compute, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/SimpleVelocityConstraint.h"
template <typename TDataType>
void declare_SimpleVelocityConstraint(py::module& m, std::string typestr) {
	using Class = dyno::SimpleVelocityConstraint<TDataType>;
	using Parent = dyno::ConstraintModule;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	typedef typename TDataType::Matrix Matrix;
	std::string pyclass_name = std::string("SimpleVelocityConstraint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("initialize", &Class::initialize)
		.def("resizeVector", &Class::resizeVector)
		.def("initialAttributes", &Class::initialAttributes)
		//.def("visValueSet", py::overload_cast<void>(&Class::visValueSet))
		.def("visValueSet", py::overload_cast<Real>(&Class::visValueSet))
		.def("visVectorSet", &Class::visVectorSet)
		.def("SIMPLE_IterNumSet", &Class::SIMPLE_IterNumSet)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inNormal", &Class::inNormal, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inSmoothingLength", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("inSamplingDistance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("varViscosity", &Class::varViscosity, py::return_value_policy::reference)
		.def("varSimpleIterationEnable", &Class::varSimpleIterationEnable, py::return_value_policy::reference);
}

#include "ParticleSystem/Module/SummationDensity.h"
template <typename TDataType>
void declare_summation_density(py::module& m, std::string typestr) {
	using Class = dyno::SummationDensity<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	std::string pyclass_name = std::string("SummationDensity") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def("compute", py::overload_cast<void>(&Class::compute))
		.def("compute", py::overload_cast<dyno::Array<Real, DeviceType::GPU>&, dyno::Array<Coord, DeviceType::GPU>&, dyno::ArrayList<int, DeviceType::GPU>&, Real, Real>(&Class::compute))
		.def("compute", py::overload_cast<dyno::Array<Real, DeviceType::GPU>&, dyno::Array<Coord, DeviceType::GPU>&, dyno::Array<Coord, DeviceType::GPU>&, dyno::ArrayList<int, DeviceType::GPU>&, Real, Real>(&Class::compute))

		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inOther", &Class::inOther, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("outDensity", &Class::outDensity, py::return_value_policy::reference)
		.def("getParticleMass", &Class::getParticleMass);
}

#include "ParticleSystem/Module/SurfaceEnergyForce.h"
template <typename TDataType>
void declare_surface_tension(py::module& m, std::string typestr) {
	using Class = dyno::SurfaceEnergyForce<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("SurfaceTension") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varKappa", &Class::varKappa, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("compute", &Class::compute);
}

#include "ParticleSystem/Module/VariationalApproximateProjection.h"
template <typename TDataType>
void declare_variational_approximate_projection(py::module& m, std::string typestr) {
	using Class = dyno::VariationalApproximateProjection<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	std::string pyclass_name = std::string("VariationalApproximateProjection") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inNormal", &Class::inNormal, py::return_value_policy::reference)
		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("compute", &Class::compute)
		.def("resizeArray", &Class::resizeArray)
		.def("varChanged", &Class::varChanged);
}

#include "ParticleSystem/ParticleSystem.h"
template <typename TDataType>
void declare_particle_system(py::module& m, std::string typestr) {
	using Class = dyno::ParticleSystem<TDataType>;
	using Parent = dyno::Node;

	class ParticleSystemTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ParticleSystem<TDataType>,
				resetStates
			);
		}

		void postUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ParticleSystem<TDataType>,
				postUpdateStates
			);
		}
	};

	class ParticleSystemPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::postUpdateStates;
	};

	std::string pyclass_name = std::string("ParticleSystem") + typestr;
	py::class_<Class, Parent, ParticleSystemTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("getNodeType", &Class::getNodeType)
		.def("statePosition", &Class::statePosition, py::return_value_policy::reference)
		.def("stateVelocity", &Class::stateVelocity, py::return_value_policy::reference)
		.def("statePointSet", &Class::statePointSet, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ParticleSystemPublicist::resetStates, py::return_value_policy::reference)
		.def("postUpdateStates", &ParticleSystemPublicist::postUpdateStates, py::return_value_policy::reference);
}

#include "ParticleSystem/ParticleFluid.h"
template <typename TDataType>
void declare_particle_fluid(py::module& m, std::string typestr) {
	using Class = dyno::ParticleFluid<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;

	class ParticleFluidTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ParticleFluid<TDataType>,
				resetStates
			);
		}

		void postUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ParticleFluid<TDataType>,
				postUpdateStates
			);
		}
	};

	class ParticleFluidPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::postUpdateStates;
	};

	std::string pyclass_name = std::string("ParticleFluid") + typestr;
	py::class_<Class, Parent, ParticleFluidTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//DEF_VAR
		.def("varSamplingDistance", &Class::varSamplingDistance, py::return_value_policy::reference)
		.def("varSmoothingLength", &Class::varSmoothingLength, py::return_value_policy::reference)
		.def("varReshuffleParticles", &Class::varReshuffleParticles, py::return_value_policy::reference)
		//DEF_NODE_PORTS
		.def("importParticleEmitters", &Class::importParticleEmitters, py::return_value_policy::reference)
		.def("getParticleEmitters", &Class::getParticleEmitters)
		.def("addParticleEmitter", &Class::addParticleEmitter)
		.def("removeParticleEmitter", &Class::removeParticleEmitter)

		.def("importInitialStates", &Class::importInitialStates, py::return_value_policy::reference)
		.def("getInitialStates", &Class::getInitialStates)
		.def("addInitialState", &Class::addInitialState)
		.def("removeInitialState", &Class::removeInitialState)

		.def("stateSamplingDistance", &Class::stateSamplingDistance, py::return_value_policy::reference)
		.def("stateSmoothingLength", &Class::stateSmoothingLength, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ParticleFluidPublicist::resetStates, py::return_value_policy::reference)
		.def("postUpdateStates", &ParticleFluidPublicist::postUpdateStates, py::return_value_policy::reference);
}

#include "ParticleSystem/GhostFluid.h"
template <typename TDataType>
void declare_ghost_fluid(py::module& m, std::string typestr) {
	using Class = dyno::GhostFluid<TDataType>;
	using Parent = dyno::ParticleFluid<TDataType>;

	class GhostFluidTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::GhostFluid<TDataType>,
				resetStates
			);
		}

		void preUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::GhostFluid<TDataType>,
				preUpdateStates
			);
		}

		void postUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::GhostFluid<TDataType>,
				postUpdateStates
			);
		}
	};

	class GhostFluidPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::preUpdateStates;
		using Class::postUpdateStates;
	};

	std::string pyclass_name = std::string("GhostFluid") + typestr;
	py::class_<Class, Parent, GhostFluidTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("statePositionMerged", &Class::statePositionMerged, py::return_value_policy::reference)
		.def("stateVelocityMerged", &Class::stateVelocityMerged, py::return_value_policy::reference)
		.def("stateAttributeMerged", &Class::stateAttributeMerged, py::return_value_policy::reference)
		.def("stateNormalMerged", &Class::stateNormalMerged, py::return_value_policy::reference)

		.def("importBoundaryParticles", &Class::importBoundaryParticles, py::return_value_policy::reference)
		.def("getBoundaryParticles", &Class::getBoundaryParticles)
		.def("addBoundaryParticle", &Class::addBoundaryParticle)
		.def("removeBoundaryParticle", &Class::removeBoundaryParticle)
		// protected
		.def("resetStates", &GhostFluidPublicist::resetStates, py::return_value_policy::reference)
		.def("preUpdateStates", &GhostFluidPublicist::preUpdateStates, py::return_value_policy::reference)
		.def("postUpdateStates", &GhostFluidPublicist::postUpdateStates, py::return_value_policy::reference);
}

#include "ParticleSystem/GhostParticles.h"
template <typename TDataType>
void declare_ghost_particles(py::module& m, std::string typestr) {
	using Class = dyno::GhostParticles<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;

	class GhostParticlesTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::GhostParticles<TDataType>,
				resetStates
			);
		}

	};

	class GhostParticlesPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("GhostParticles") + typestr;
	py::class_<Class, Parent, GhostParticlesTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("stateNormal", &Class::stateNormal, py::return_value_policy::reference)
		.def("stateAttribute", &Class::stateAttribute, py::return_value_policy::reference)
		// protected
		.def("resetStates", &GhostParticlesPublicist::resetStates, py::return_value_policy::reference);
}

#include "ParticleSystem/MakeGhostParticles.h"
template <typename TDataType>
void declare_make_ghost_particles(py::module& m, std::string typestr) {
	using Class = dyno::MakeGhostParticles<TDataType>;
	using Parent = dyno::GhostParticles<TDataType>;

	class MakeGhostParticlesTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MakeGhostParticles<TDataType>,
				resetStates
			);
		}

	};

	class MakeGhostParticlesPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("MakeGhostParticles") + typestr;
	py::class_<Class, Parent, MakeGhostParticlesTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inPoints", &Class::inPoints, py::return_value_policy::reference)
		.def("varReverseNormal", &Class::varReverseNormal, py::return_value_policy::reference)
		// protected
		.def("resetStates", &MakeGhostParticlesPublicist::resetStates, py::return_value_policy::reference);
}

#include "ParticleSystem/MakeParticleSystem.h"
template <typename TDataType>
void declare_make_particle_system(py::module& m, std::string typestr) {
	using Class = dyno::MakeParticleSystem<TDataType>;
	using Parent = dyno::ParticleSystem<TDataType>;

	class MakeParticleSystemTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::MakeParticleSystem<TDataType>,
				resetStates
			);
		}

	};

	class MakeParticleSystemPublicist : public Class
	{
	public:
		using Class::resetStates;
	};

	std::string pyclass_name = std::string("MakeParticleSystem") + typestr;
	py::class_<Class, Parent, MakeParticleSystemTrampoline, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varInitialVelocity", &Class::varInitialVelocity, py::return_value_policy::reference)
		.def("inPoints", &Class::inPoints, py::return_value_policy::reference)
		// protected
		.def("resetStates", &MakeParticleSystemPublicist::resetStates, py::return_value_policy::reference);
}

#include "ParticleSystem/ParticleSystemHelper.h"
template <typename TDataType>
void declare_particle_system_helper(py::module& m, std::string typestr) {
	using Class = dyno::ParticleSystemHelper<TDataType>;
	std::string pyclass_name = std::string("ParticleSystemHelper") + typestr;
	py::class_<Class, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("calculateMortonCode", &Class::calculateMortonCode)
		.def("reorderParticles", &Class::reorderParticles);
}

void pybind_particle_system(py::module& m);