/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-06
 * @description: Implemendation of SemiAnalyticalIncompressibleFluidModel class, a container for semi-analytical projection-based fluids 
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */
#include "SemiAnalyticalIncompressibleFluidModel.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"

#include "Collision/NeighborPointQuery.h"
#include "Collision/NeighborTriangleQuery.h"

#include "SemiAnalyticalIncompressibilityModule.h"
#include "Node.h"

namespace dyno {

	IMPLEMENT_TCLASS(SemiAnalyticalIncompressibleFluidModel, TDataType)

	template <typename TDataType>
	SemiAnalyticalIncompressibleFluidModel<TDataType>::SemiAnalyticalIncompressibleFluidModel()
		: GroupModule()
		, m_restRho(Real(1000))
		, m_pNum(0)
	{
		m_smoothing_length.setValue(Real(0.015));

		m_nbrQueryPoint = std::make_shared<NeighborPointQuery<TDataType>>();
		m_smoothing_length.connect(m_nbrQueryPoint->inRadius());
		m_particle_position.connect(m_nbrQueryPoint->inPosition());

		//m_nbrQueryPoint->initialize();

		m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		m_particle_position.connect(m_integrator->inPosition());
		m_particle_velocity.connect(m_integrator->inVelocity());
		m_particle_force_density.connect(m_integrator->inForceDensity());
		//m_integrator->initialize();

		m_nbrQueryTri = std::make_shared<NeighborTriangleQuery<TDataType>>();
		m_smoothing_length.connect(m_nbrQueryTri->inRadius());
		this->m_particle_position.connect(m_nbrQueryTri->inPosition());
		this->m_triangle_vertex.connect(m_nbrQueryTri->inTriPosition());
		this->m_triangle_index.connect(m_nbrQueryTri->inTriangles());

		//m_nbrQueryTri->initialize();

		m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		m_visModule->varViscosity()->setValue(Real(1));
		m_smoothing_length.connect(m_visModule->inSmoothingLength());
		this->m_particle_position.connect(m_visModule->inPosition());
		this->m_particle_velocity.connect(m_visModule->inVelocity());
		m_nbrQueryPoint->outNeighborIds()->connect(m_visModule->inNeighborIds());
		//m_visModule->initialize();

		printf("finished initialize fluid viscosity\n");

		m_velocity_mod.resize(m_particle_velocity.size());
		m_flip.resize(m_particle_velocity.size());

		m_meshCollision = std::make_shared<TriangularMeshConstraint<TDataType>>();
		this->m_particle_position.connect(m_meshCollision->inPosition());
		this->m_particle_velocity.connect(m_meshCollision->inVelocity());
		this->m_triangle_vertex.connect(m_meshCollision->inTriangleVertex());
		this->m_triangle_index.connect(m_meshCollision->inTriangleIndex());
		m_nbrQueryTri->outNeighborIds()->connect(m_meshCollision->inTriangleNeighborIds());
		//pReduce = Reduction<Real>::Create(m_velocity_mod.size());

		printf("finished initialize mesh collision\n");

		m_pbdModule = std::make_shared<SemiAnalyticalIncompressibilityModule<TDataType>>();
		m_smoothing_length.connect(&m_pbdModule->m_smoothing_length);

		m_particle_velocity.connect(&m_pbdModule->m_particle_velocity);

		m_particle_mass.connect(&m_pbdModule->m_particle_mass);
		this->m_particle_attribute.connect(&m_pbdModule->m_particle_attribute);

		this->m_particle_position.connect(&m_pbdModule->m_particle_position);

		this->m_triangle_vertex.connect(&m_pbdModule->m_triangle_vertex);
		this->m_triangle_vertex_old.connect(&m_pbdModule->m_triangle_vertex_old);

		this->m_triangle_vertex_mass.connect(&m_pbdModule->m_triangle_vertex_mass);

		this->m_triangle_index.connect(&m_pbdModule->m_triangle_index);

		m_nbrQueryPoint->outNeighborIds()->connect(m_pbdModule->inNeighborParticleIds());
		m_flip.connect(&m_pbdModule->m_flip);
		m_nbrQueryTri->outNeighborIds()->connect(m_pbdModule->inNeighborTriangleIds());
		m_pbdModule->initialize();
	}

	template <typename TDataType>
	void SemiAnalyticalIncompressibleFluidModel<TDataType>::updateImpl()
	{
		//return;
		//         if (first == 1)
		//         {
		//
		//             m_nbrQueryTri = this->getParent()->addComputeModule<NeighborQuery<TDataType>>("neighborhood3");
		//             m_smoothing_length.connect(m_nbrQueryTri->m_radius);
		//             this->m_particle_position.connect(m_nbrQueryTri->m_position);
		//             this->m_triangle_vertex.connect(m_nbrQueryTri->m_TriPos);
		//             this->m_triangle_index.connect(m_nbrQueryTri->m_triangls);
		//             m_nbrQueryTri->initialize();
		//             m_nbrQueryTri->m_neighborhood.connect(m_pbdModule->m_neighborhood_triangles);
		//
		//             m_pbdModule->initialize_();
		//             first = 0;
		//             return;
		//
		//         }

		m_velocity_mod.resize(m_particle_velocity.size());
		m_flip.resize(m_particle_velocity.size());
		m_flip.getData().reset();

		//m_integrator->begin();

		m_integrator->integrate();

		m_nbrQueryPoint->compute();

		m_nbrQueryTri->compute();

		m_meshCollision->update();

		m_visModule->constrain();

		m_pbdModule->constrain();

		m_integrator->end();
	}

	DEFINE_CLASS(SemiAnalyticalIncompressibleFluidModel);
}  // namespace dyno