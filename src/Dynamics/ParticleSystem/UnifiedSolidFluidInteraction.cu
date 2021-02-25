#include "UnifiedSolidFluidInteraction.h"
#include "PositionBasedFluidModel.h"

#include "Topology/PointSet.h"
//#include "PointRenderModule.h"
#include "Utility.h"
#include "ParticleSystem.h"
#include "Topology/NeighborQuery.h"
#include "Kernel.h"
#include "DensityPBD.h"
#include "ImplicitViscosity.h"
#include "Attribute.h"
#include "TriangularSurfaceMeshNode.h"
#include "UnifiedVelocityNode.h"
#include "PositionBasedFluidModelMesh.h"
#include "UnifiedFluidRigidConstraint.h"
#include "RigidBody/RigidBodySystem.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(UnifiedSolidFluidInteraction, TDataType)


	template<typename TDataType>
	UnifiedSolidFluidInteraction<TDataType>::UnifiedSolidFluidInteraction(std::string name)
		:Node(name)
	{
		this->attachField(&radius, "radius", "radius");
		radius.setValue(0.0065);



		
		

		//m_rigidModule


		//this->setNumericalModel(pbd);
	}


	template<typename TDataType>
	void UnifiedSolidFluidInteraction<TDataType>::setInteractionDistance(Real d)
	{
		radius.setValue(d);
	}

	template<typename TDataType>
	UnifiedSolidFluidInteraction<TDataType>::~UnifiedSolidFluidInteraction()
	{

	}

	template<typename TDataType>
	bool UnifiedSolidFluidInteraction<TDataType>::initialize()
	{
		return true;
	}



	template<typename TDataType>
	bool UnifiedSolidFluidInteraction<TDataType>::resetStatus()
	{
		printf("sfi~ initialize\n");
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems = this->getParticleSystems();
		std::vector<std::shared_ptr<TriangularSurfaceMeshNode<TDataType>>> m_surfaces = this->getTriangularSurfaceMeshNodes();

		int total_num = 0;

		std::vector<int> ids;
		std::vector<int> fxd;
		std::vector<Real> mass;
		std::vector<Real> mass_tri;
		for (int i = 0; i < m_particleSystems.size(); i++)
		{
			printf("%s\n", m_particleSystems[i]->getName().c_str());
			if (!(m_particleSystems[i]->getName().c_str()[0] == 'f')) // solid
			{
				continue;

			}

			m_particleSystems[i]->self_update = false;

			if (m_particleSystems[i]->currentPosition()->getElementCount() == 0)
			{
				//printf("out0\n");
				continue;
			}
			auto points = m_particleSystems[i]->currentPosition()->getValue();

			//printf("out0\n");

			total_num += points.size();



			Real m = m_particleSystems[i]->getMass() * 10.0;// / points.size() * 100000.0;

			for (int j = 0; j < points.size(); j++)
			{
				ids.push_back(i);
				mass.push_back(m);
				fxd.push_back((0));
			}
			printf("out\n");
		}
		printf("%d\n", total_num);

		if (total_num > 0)
			m_objId.resize(total_num);

		m_particle_velocity.setElementCount(total_num);
		m_particle_mass.setElementCount(total_num);
		m_particle_position.setElementCount(total_num);
		m_particle_force_density.setElementCount(total_num);
		//		m_fixed.setElementCount(total_num);
		ParticleId.setElementCount(total_num);
		//		ElasityPressure.setElementCount(total_num);
		if (total_num > 0)
		{
			posBuf.resize(total_num);
			weights.resize(total_num);
			init_pos.resize(total_num);
			VelBuf.resize(total_num);

			Function1Pt::copy(m_objId, ids);
			Function1Pt::copy(ParticleId.getValue(), ids);
		}
		fxd.clear();
		ids.clear();
		mass.clear();
		mass_tri.clear();

		m_particle_attribute.setElementCount(total_num);

		//		m_normal.setElementCount(total_num);

		if (total_num > 0)
		{
			int start = 0;
			DeviceArray<Coord>& allpoints = m_particle_position.getValue();
			DeviceArray<Attribute>& allattrs = m_particle_attribute.getValue();


			std::vector<Attribute> attributeList;
			int sumTri = 0;
			int offset = 0;

			for (int i = 0; i < m_particleSystems.size(); i++)
			{
				if (m_particleSystems[i]->currentPosition()->getElementCount() == 0)
				{
					continue;
				}
				DeviceArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
				DeviceArray<Coord>& vels = m_particleSystems[i]->currentVelocity()->getValue();

				int num = points.size();
				//printf("%s\n",m_particleSystems[i]->getName().c_str());
				bool fixx = false;

				if (!(m_particleSystems[i]->getName().c_str()[0] == 'f')) // solid
				{
					continue;

				}
				else//fluid
				{
					for (int j = 0; j < num; j++)
					{
						Attribute a = Attribute();
						a.SetFluid();
						a.SetDynamic();
						attributeList.push_back(a);
					}

				}


				cudaMemcpy(allpoints.begin() + start, points.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				cudaMemcpy(m_particle_velocity.getValue().begin() + start, vels.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);

				start += num;
			}
			Function1Pt::copy(m_particle_attribute.getValue(), attributeList);
			attributeList.clear();
		}



		int total_tri = 0;
		int total_tri_pos = 0;
		for (int i = 0; i < m_surfaces.size(); i++)
		{
			total_tri += m_surfaces[i]->getTriangleIndex()->getReference()->size();
			total_tri_pos += m_surfaces[i]->getVertexPosition()->getReference()->size();
		}

		m_triangle_vertex_mass.setElementCount(total_tri_pos);
		m_triangle_vertex.setElementCount(total_tri_pos);
		m_triangle_vertex_old.setElementCount(total_tri_pos);

		m_triangle_index.setElementCount(total_tri);
		printf("**********************TRI: %d\n", total_tri);
		int start_vertex = 0;
		int start_triangle = 0;
		for (int i = 0; i < m_surfaces.size(); i++)
		{

			auto vertex_position = m_surfaces[i]->getVertexPosition()->getValue();///!!!
			int num_vertex = vertex_position.size();

			printf("mesh vertex size: %d\n", num_vertex);


			cudaMemcpy(m_triangle_vertex.getValue().begin() + start_vertex, vertex_position.begin(), num_vertex * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(m_triangle_vertex_old.getValue().begin() + start_vertex, vertex_position.begin(), num_vertex * sizeof(Coord), cudaMemcpyDeviceToDevice);

			auto triangle_index = m_surfaces[i]->getTriangleIndex()->getValue();
			int num_triangle = triangle_index.size();

			HostArray<Triangle> host_triangle;
			host_triangle.resize(num_triangle);

			printf("mesh tri size: %d %d %d\n", num_triangle, start_triangle, start_vertex);

			Function1Pt::copy(host_triangle, triangle_index);
			for (int j = 0; j < num_triangle; j++)
			{
				host_triangle[j][0] += start_vertex;
				host_triangle[j][1] += start_vertex;
				host_triangle[j][2] += start_vertex;
			}
			//printf("%d\n", start_vertex);
			cudaMemcpy(m_triangle_index.getValue().begin() + start_triangle, host_triangle.begin(), num_triangle * sizeof(Triangle), cudaMemcpyHostToDevice);

			host_triangle.release();

			start_vertex += num_vertex;
			start_triangle += num_triangle;
		}


		printf("INSIDE UNIFIED SFI INITIALIZE NEW\n");
		m_rigid_mass.connect(m_rigidModule->currentMass());
		m_rigid_interior.connect(&m_rigidModule->m_inertia);
		m_rigid_velocity.connect(m_rigidModule->currentVelocity());
		m_rigid_angular_velocity.connect(m_rigidModule->currentAngularVelocity());
		m_rigid_position.connect(m_rigidModule->currentCenter());
		m_rigid_rotation.connect(m_rigidModule->currentRotation());
		AA.connect(&m_rigidModule->AA);
		m_rigidModule->late_initialize = false;
		m_rigidModule->initialize();



		m_fluidModule = std::make_shared<UnifiedVelocityNode<DataType3f>>();

		m_particle_position.connect(&m_fluidModule->m_particle_position);
		m_particle_velocity.connect(&m_fluidModule->m_particle_velocity);
		m_particle_force_density.connect(&m_fluidModule->m_particle_force_density);
		m_particle_attribute.connect(&m_fluidModule->m_particle_attribute);
		m_particle_mass.connect(&m_fluidModule->m_particle_mass);
		m_triangle_vertex_mass.connect(&m_fluidModule->m_triangle_vertex_mass);
		m_triangle_index.connect(&m_fluidModule->m_triangle_index);
		m_triangle_vertex.connect(&m_fluidModule->m_triangle_vertex);
		m_triangle_vertex_old.connect(&m_fluidModule->m_triangle_vertex_old);

		m_boundary_pressure.connect(&m_fluidModule->m_force_rigid);
		m_gradient_point.connect(&m_fluidModule->m_gradient_point);
		m_nbrcons.connect(&m_fluidModule->m_nbrcons); ///!!!!!!!!!!!!!!!!!!!
		m_gradient_rigid.connect(&m_fluidModule->m_gradient_rigid);
		m_velocity_inside_iteration.connect(&m_fluidModule->m_velocity_inside_iteration);

		m_fluidModule->setSmoothingLength(0.0075);
		this->setNumericalModel(m_fluidModule);//m_fluidModule->initialize();


		m_intermediateModule = std::make_shared<UnifiedFluidRigidConstraint<DataType3f>>();

		m_boundary_pressure.connect(&m_intermediateModule->m_boundary_forces);
		m_gradient_point.connect(&m_intermediateModule->m_gradient_point);
		//m_nbrcons.connect(&m_intermediateModule->m_nbr_cons); ///!!!!!!!!!!!!!!!!!!!
		m_gradient_rigid.connect(&m_intermediateModule->m_gradient_boundary);
		m_velocity_inside_iteration.connect(&m_intermediateModule->m_fluid_tmp_vel);
		m_rigid_mass.connect(&m_intermediateModule->m_rigid_mass);
		m_rigid_interior.connect(&m_intermediateModule->m_rigid_interior);
		m_rigid_velocity.connect(&m_intermediateModule->m_rigid_velocity);
		m_rigid_angular_velocity.connect(&m_intermediateModule->m_rigid_angular_velocity);
		m_rigid_position.connect(&m_intermediateModule->m_rigid_position);
		m_rigid_rotation.connect(&m_intermediateModule->m_rigid_rotation);
		AA.connect(&m_intermediateModule->AA);
		m_particle_position.connect(&m_intermediateModule->m_particle_position);
		m_particle_velocity.connect(&m_intermediateModule->m_particle_velocity);
		
		auto discreteSet = TypeInfo::cast<DiscreteElements<DataType3f>>(m_rigidModule->getTopologyModule());
		m_intermediateModule->setDiscreteSet(discreteSet);

		m_intermediateModule->initialize();

		return true;
	}

	template<typename TDataType>
	void UnifiedSolidFluidInteraction<TDataType>::advance(Real dt)
	{
		//return;
		printf("inside advance\n");
		//return;
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems = this->getParticleSystems();
		std::vector<std::shared_ptr<TriangularSurfaceMeshNode<TDataType>>> m_surfaces = this->getTriangularSurfaceMeshNodes();
		

		//auto nModel = this->getNumericalModel();
		
		{
			//printf("+++++++++++++++++++++++++++++++++++++++++++++++++++ %d %d\n", m_particle_position.getElementCount(), m_particle_velocity.getElementCount());
			//nModel->step(this->getDt());
			
			m_rigidModule->pretreat(dt);
			m_fluidModule->pretreat(dt);
			m_intermediateModule->pretreat(dt);

			m_nbrcons.setElementCount(m_intermediateModule->m_nbr_cons.getElementCount());
			if(m_intermediateModule->m_nbr_cons.getElementCount() > 0)
				Function1Pt::copy(m_nbrcons.getValue(), m_intermediateModule->m_nbr_cons.getValue());

			for (int i = 0; i < 25; i++)
			{
				m_rigidModule->take_one_iteration(dt);
				m_fluidModule->take_one_iteration1(dt);
				m_intermediateModule->take_one_iteration_1(dt);
				m_fluidModule->take_one_iteration2(dt);
				m_intermediateModule->take_one_iteration_2(dt);
			}

			m_rigidModule->update(dt);
			m_fluidModule->update(dt);
			m_intermediateModule->update(dt);
		}
		
		if (m_particle_position.getElementCount() > 0)
		{
			int start = 0;
			DeviceArray<Coord>& allvels = m_particle_velocity.getValue();
			DeviceArray<Coord>& allposs = m_particle_position.getValue();

			for (int i = 0; i < m_particleSystems.size(); i++)
			{
				if (!(m_particleSystems[i]->getName().c_str()[0] == 'f')) // solid
				{
					continue;

				}
				if (m_particleSystems[i]->currentPosition()->getElementCount() == 0)
				{
					continue;
				}
				DeviceArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
				DeviceArray<Coord>& vels = m_particleSystems[i]->currentVelocity()->getValue();
				int num = points.size();
				cudaMemcpy(points.begin(), allposs.begin() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				cudaMemcpy(vels.begin(), allvels.begin() + start, num * sizeof(Coord), cudaMemcpyDeviceToDevice);

				start += num;
			}
		}
		int start_vertex = 0;
		int start_triangle = 0;

		cudaMemcpy(m_triangle_vertex_old.getValue().begin(), m_triangle_vertex.getValue().begin(), m_triangle_vertex.getElementCount() * sizeof(Coord), cudaMemcpyDeviceToDevice);

		for (int i = 0; i < m_surfaces.size(); i++)
		{

			auto vertex_position = m_surfaces[i]->getVertexPosition()->getValue();
			int num_vertex = vertex_position.size();

			cudaMemcpy(m_triangle_vertex.getValue().begin() + start_vertex, vertex_position.begin(), num_vertex * sizeof(Coord), cudaMemcpyDeviceToDevice);


			start_vertex += num_vertex;
		}
		

	}
}