#include "../PyCommon.h"

#include "SemiAnalyticalScheme/ComputeParticleAnisotropy.h"
template <typename TDataType>
void declare_compute_particle_anisotropy(py::module& m, std::string typestr) {
	using Class = dyno::ComputeParticleAnisotropy<TDataType>;
	using Parent = dyno::ComputeModule;
	std::string pyclass_name = std::string("ComputeParticleAnisotropy") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("compute", &Class::compute)
		.def("varSmoothingLength", &Class::varSmoothingLength, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("outTransform", &Class::outTransform, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/ParticleRelaxtionOnMesh.h"
template <typename TDataType>
void declare_particle_relaxtion_on_mesh(py::module& m, std::string typestr) {
	using Class = dyno::ParticleRelaxtionOnMesh<TDataType>;
	using Parent = dyno::PointsBehindMesh<TDataType>;

	class ParticleRelaxtionOnMeshTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::ParticleRelaxtionOnMesh<TDataType>,
				resetStates,
			);
		}
	};

	class ParticleRelaxtionOnMeshPublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::preUpdateStates;
		using Class::particleRelaxion;
		using Class::updatePositions;
	};

	std::string pyclass_name = std::string("ParticleRelaxtionOnMesh") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varIterationNumber", &Class::varIterationNumber, py::return_value_policy::reference)
		.def("varNormalForceStrength", &Class::varNormalForceStrength, py::return_value_policy::reference)
		.def("varMeshCollisionThickness", &Class::varMeshCollisionThickness, py::return_value_policy::reference)
		.def("varPointNeighborLength", &Class::varPointNeighborLength, py::return_value_policy::reference)
		.def("varMeshNeighborLength", &Class::varMeshNeighborLength, py::return_value_policy::reference)
		.def("varViscosityStrength", &Class::varViscosityStrength, py::return_value_policy::reference)
		.def("stateDelta", &Class::stateDelta, py::return_value_policy::reference)
		.def("varDensityIteration", &Class::varDensityIteration, py::return_value_policy::reference)
		// protected
		.def("resetStates", &ParticleRelaxtionOnMeshPublicist::resetStates, py::return_value_policy::reference)
		.def("preUpdateStates", &ParticleRelaxtionOnMeshPublicist::preUpdateStates, py::return_value_policy::reference)
		.def("particleRelaxion", &ParticleRelaxtionOnMeshPublicist::particleRelaxion, py::return_value_policy::reference)
		.def("updatePositions", &ParticleRelaxtionOnMeshPublicist::updatePositions, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalIncompressibilityModule.h"
template <typename TDataType>
void declare_semi_analytical_incompressibility_module(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalIncompressibilityModule<TDataType>;
	using Parent = dyno::ConstraintModule;

	class SemiAnalyticalIncompressibilityModuleTrampoline : public Class
	{
	public:
		using Class::Class;

		bool initializeImpl() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::SemiAnalyticalIncompressibilityModule<TDataType>,
				initializeImpl,
				);
		}
	};

	class SemiAnalyticalIncompressibilityModulePublicist : public Class
	{
	public:
		using Class::initializeImpl;
	};

	std::string pyclass_name = std::string("SemiAnalyticalIncompressibilityModule") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)
		.def("getPosition", &Class::getPosition)
		.def("inNeighborParticleIds", &Class::inNeighborParticleIds, py::return_value_policy::reference)
		.def("inNeighborTriangleIds", &Class::inNeighborTriangleIds, py::return_value_policy::reference)
		.def_readwrite("m_smoothing_length", &Class::m_smoothing_length)
		.def_readwrite("m_sampling_distance", &Class::m_sampling_distance)
		.def_readwrite("m_particle_mass", &Class::m_particle_mass)
		.def_readwrite("m_particle_position", &Class::m_particle_position)
		.def_readwrite("m_particle_velocity", &Class::m_particle_velocity)
		.def_readwrite("m_particle_attribute", &Class::m_particle_attribute)
		.def_readwrite("m_flip", &Class::m_flip)
		.def_readwrite("m_triangle_vertex_mass", &Class::m_triangle_vertex_mass)
		.def_readwrite("m_triangle_vertex", &Class::m_triangle_vertex)
		.def_readwrite("m_triangle_vertex_old", &Class::m_triangle_vertex_old)
		.def_readwrite("m_triangle_index", &Class::m_triangle_index)
		// protected
		.def("initializeImpl", &SemiAnalyticalIncompressibilityModulePublicist::initializeImpl, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalIncompressibleFluidModel.h"
template <typename TDataType>
void declare_semi_analytical_incompressible_fluid_model(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalIncompressibleFluidModel<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("SemiAnalyticalIncompressibleFluidModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("updateImpl", &Class::updateImpl)
		.def("setSmoothingLength", &Class::setSmoothingLength)
		.def("setRestDensity", &Class::setRestDensity)
		.def_readwrite("m_smoothing_length", &Class::m_smoothing_length)
		.def_readwrite("max_vel", &Class::max_vel)
		.def_readwrite("var_smoothing_length", &Class::var_smoothing_length)
		.def_readwrite("m_particle_mass", &Class::m_particle_mass)
		.def_readwrite("m_particle_position", &Class::m_particle_position)
		.def_readwrite("m_particle_velocity", &Class::m_particle_velocity)
		.def_readwrite("m_particle_attribute", &Class::m_particle_attribute)
		.def_readwrite("m_triangle_vertex_mass", &Class::m_triangle_vertex_mass)
		.def_readwrite("m_triangle_vertex", &Class::m_triangle_vertex)
		.def_readwrite("m_triangle_vertex_old", &Class::m_triangle_vertex_old)
		.def_readwrite("m_triangle_index", &Class::m_triangle_index)
		.def_readwrite("m_particle_force_density", &Class::m_particle_force_density)
		.def_readwrite("m_vertex_force_density", &Class::m_vertex_force_density)
		.def_readwrite("m_vn", &Class::m_vn)
		.def_readwrite("m_flip", &Class::m_flip)
		.def_readwrite("pReduce", &Class::pReduce)
		.def_readwrite("m_velocity_mod", &Class::m_velocity_mod);
}

#include "SemiAnalyticalScheme/SemiAnalyticalParticleShifting.h"
template <typename TDataType>
void declare_semi_analytical_particle_shifting(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalParticleShifting<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;

	class SemiAnalyticalParticleShiftingTrampoline : public Class
	{
	public:
		using Class::Class;

		void compute() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SemiAnalyticalParticleShifting<TDataType>,
				compute,
				);
		}
	};

	class SemiAnalyticalParticleShiftingPublicist : public Class
	{
	public:
		using Class::compute;
	};

	std::string pyclass_name = std::string("SemiAnalyticalParticleShifting") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varInterationNumber", &Class::varInterationNumber, py::return_value_policy::reference)
		.def("varInertia", &Class::varInertia, py::return_value_policy::reference)
		.def("varBulk", &Class::varBulk, py::return_value_policy::reference)
		.def("varSurfaceTension", &Class::varSurfaceTension, py::return_value_policy::reference)
		.def("varAdhesionIntensity", &Class::varAdhesionIntensity, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)

		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("inNeighborTriIds", &Class::inNeighborTriIds, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		// protected
		.def("compute", &SemiAnalyticalParticleShiftingPublicist::compute, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalPBD.h"
template <typename TDataType>
void declare_semi_analytical_pbd(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalPBD<TDataType>;
	using Parent = dyno::ConstraintModule;
	std::string pyclass_name = std::string("SemiAnalyticalPBD") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("constrain", &Class::constrain)

		.def("varInterationNumber", &Class::varInterationNumber, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inSmoothingLength", &Class::inSmoothingLength, py::return_value_policy::reference)
		.def("inSamplingDistance", &Class::inSamplingDistance, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("inNeighborParticleIds", &Class::inNeighborParticleIds, py::return_value_policy::reference)
		.def("in_neighbor_triang_ids", &Class::inNeighborTriangleIds, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalPositionBasedFluidModel.h"
template <typename TDataType>
void declare_semi_analytical_position_based_fluid_model(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalPositionBasedFluidModel<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("SemiAnalyticalPositionBasedFluidModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSmoothingLength", &Class::varSmoothingLength, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalSFINode.h"
template <typename TDataType>
void declare_semi_analytical_sfi_node(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalSFINode<TDataType>;
	using Parent = dyno::ParticleFluid<TDataType>;

	class SemiAnalyticalSFINodeTrampoline : public Class
	{
	public:
		using Class::Class;

		void resetStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SemiAnalyticalSFINode<TDataType>,
				resetStates,
				);
		}

		void preUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SemiAnalyticalSFINode<TDataType>,
				preUpdateStates,
				);
		}

		void postUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::SemiAnalyticalSFINode<TDataType>,
				postUpdateStates,
				);
		}

		bool validateInputs() override
		{
			PYBIND11_OVERRIDE(
				bool,
				dyno::SemiAnalyticalSFINode<TDataType>,
				validateInputs,
				);
		}
	};

	class SemiAnalyticalSFINodePublicist : public Class
	{
	public:
		using Class::resetStates;
		using Class::preUpdateStates;
		using Class::postUpdateStates;
		using Class::validateInputs;
	};

	std::string pyclass_name = std::string("SemiAnalyticalSFINode") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("varFast", &Class::varFast, py::return_value_policy::reference)
		.def("varSyncBoundary", &Class::varSyncBoundary, py::return_value_policy::reference)
		// protected
		.def("resetStates", &SemiAnalyticalSFINodePublicist::resetStates, py::return_value_policy::reference)
		.def("preUpdateStates", &SemiAnalyticalSFINodePublicist::preUpdateStates, py::return_value_policy::reference)
		.def("postUpdateStates", &SemiAnalyticalSFINodePublicist::postUpdateStates, py::return_value_policy::reference)
		.def("validateInputs", &SemiAnalyticalSFINodePublicist::validateInputs, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalSummationDensity.h"
#include "Array/Array.h"
template <typename TDataType>
void declare_semi_analytical_summation_density(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalSummationDensity<TDataType>;
	using Parent = dyno::ParticleApproximation<TDataType>;
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;
	std::string pyclass_name = std::string("SemiAnalyticalSummationDensity") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		//.def("compute", py::overload_cast<void>(&Class::compute))
		.def("compute", py::overload_cast<dyno::Array<Real, DeviceType::GPU>&, dyno::Array<Coord, DeviceType::GPU>&, dyno::Array<dyno::TopologyModule::Triangle, DeviceType::GPU>&, dyno::Array<Coord, DeviceType::GPU>&, dyno::ArrayList<int, DeviceType::GPU>&, dyno::ArrayList<int, DeviceType::GPU>&, Real, Real, Real>(&Class::compute))

		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inNeighborIds", &Class::inNeighborIds, py::return_value_policy::reference)
		.def("inNeighborTriIds", &Class::inNeighborTriIds, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("outDensity", &Class::outDensity, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/SemiAnalyticalSurfaceTensionModel.h"
template <typename TDataType>
void declare_semi_analytical_surface_tension_model(py::module& m, std::string typestr) {
	using Class = dyno::SemiAnalyticalSurfaceTensionModel<TDataType>;
	using Parent = dyno::GroupModule;
	std::string pyclass_name = std::string("SemiAnalyticalSurfaceTensionModel") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varSmoothingLength", &Class::varSmoothingLength, py::return_value_policy::reference)
		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)

		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inForceDensity", &Class::inForceDensity, py::return_value_policy::reference)

		.def("inAttribute", &Class::inAttribute, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)

		.def("varSurfaceTension", &Class::varSurfaceTension, py::return_value_policy::reference)
		.def("varAdhesionIntensity", &Class::varAdhesionIntensity, py::return_value_policy::reference)
		.def("varRestDensity", &Class::varRestDensity, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/TriangularMeshBoundary.h"
template <typename TDataType>
void declare_triangular_mesh_boundary(py::module& m, std::string typestr) {
	using Class = dyno::TriangularMeshBoundary<TDataType>;
	using Parent = dyno::Node;

	class TriangularMeshBoundaryTrampoline : public Class
	{
	public:
		using Class::Class;

		void preUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TriangularMeshBoundary<TDataType>,
				preUpdateStates,
				);
		}

		void updateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TriangularMeshBoundary<TDataType>,
				updateStates,
				);
		}

		void postUpdateStates() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TriangularMeshBoundary<TDataType>,
				postUpdateStates,
				);
		}

	};

	class TriangularMeshBoundaryPublicist : public Class
	{
	public:
		using Class::preUpdateStates;
		using Class::updateStates;
		using Class::postUpdateStates;
	};

	std::string pyclass_name = std::string("TriangularMeshBoundary") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varThickness", &Class::varThickness, py::return_value_policy::reference)
		.def("varTangentialFriction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("varNormalFriction", &Class::varNormalFriction, py::return_value_policy::reference)

		.def("get_particle_system", &Class::getParticleSystems)
		.def("addParticleSystem", &Class::addParticleSystem)
		.def("removeParticleSystem", &Class::removeParticleSystem)
		.def("importParticleSystems", &Class::importParticleSystems, py::return_value_policy::reference)

		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)

		.def("statePosition", &Class::statePosition, py::return_value_policy::reference)
		.def("stateVelocity", &Class::stateVelocity, py::return_value_policy::reference)
		// protected
		.def("updateStates", &TriangularMeshBoundaryPublicist::updateStates, py::return_value_policy::reference)
		.def("preUpdateStates", &TriangularMeshBoundaryPublicist::preUpdateStates, py::return_value_policy::reference)
		.def("postUpdateStates", &TriangularMeshBoundaryPublicist::postUpdateStates, py::return_value_policy::reference);
}

#include "SemiAnalyticalScheme/TriangularMeshConstraint.h"
template <typename TDataType>
void declare_triangular_mesh_constraint(py::module& m, std::string typestr) {
	using Class = dyno::TriangularMeshConstraint<TDataType>;
	using Parent = dyno::ConstraintModule;

	class TriangularMeshConstraintTrampoline : public Class
	{
	public:
		using Class::Class;

		void constrain() override
		{
			PYBIND11_OVERRIDE(
				void,
				dyno::TriangularMeshConstraint<TDataType>,
				constrain,
				);
		}

	};

	class TriangularMeshConstraintPublicist : public Class
	{
	public:
		using Class::constrain;
	};

	std::string pyclass_name = std::string("TriangularMeshConstraint") + typestr;
	py::class_<Class, Parent, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>())
		.def("varThickness", &Class::varThickness, py::return_value_policy::reference)
		.def("varTangentialFriction", &Class::varTangentialFriction, py::return_value_policy::reference)
		.def("varNormalFriction", &Class::varNormalFriction, py::return_value_policy::reference)

		.def("inTimeStep", &Class::inTimeStep, py::return_value_policy::reference)
		.def("inPosition", &Class::inPosition, py::return_value_policy::reference)
		.def("inVelocity", &Class::inVelocity, py::return_value_policy::reference)
		.def("inTriangleSet", &Class::inTriangleSet, py::return_value_policy::reference)
		.def("inTriangleNeighborIds", &Class::inTriangleNeighborIds, py::return_value_policy::reference)
		// protected
		.def("constrain", &TriangularMeshConstraintPublicist::constrain, py::return_value_policy::reference);
}

void pybind_semi_analytical_scheme(py::module& m);