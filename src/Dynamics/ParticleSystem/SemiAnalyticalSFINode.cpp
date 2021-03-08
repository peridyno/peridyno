#include "SemiAnalyticalSFINode.h"
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
#include "SemiAnalyticalIncompressibleFluidModel.h"
#include "PositionBasedFluidModelMesh.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(SemiAnalyticalSFINode, TDataType)

	template<typename TDataType>
	SemiAnalyticalSFINode<TDataType>::SemiAnalyticalSFINode(std::string name)
		:Node(name)
	{
		this->attachField(&radius, "radius", "radius");
		radius.setValue(0.0125);

		/*
		auto pbd = std::make_shared<SemiAnalyticalIncompressibleFluidModel<DataType3f>>();

		m_particle_position.connect(&pbd->m_particle_position);
		m_particle_velocity.connect(&pbd->m_particle_velocity);
		m_particle_force_density.connect(&pbd->m_particle_force_density);
		m_particle_attribute.connect(&pbd->m_particle_attribute);
		
		m_particle_mass.connect(&pbd->m_particle_mass);

		
		m_triangle_vertex_mass.connect(&pbd->m_triangle_vertex_mass);
		m_triangle_index.connect(&pbd->m_triangle_index);
		m_triangle_vertex.connect(&pbd->m_triangle_vertex);
		m_triangle_vertex_old.connect(&pbd->m_triangle_vertex_old);
		pbd->setSmoothingLength(0.0125);
		*/
		
		
		std::shared_ptr<PositionBasedFluidModelMesh<DataType3f>> pbd = std::make_shared<PositionBasedFluidModelMesh<DataType3f>>();
		m_particle_position.connect(&pbd->m_position);
		m_particle_velocity.connect(&pbd->m_velocity);
		m_particle_force_density.connect(&pbd->m_forceDensity);
		m_particle_mass.connect(&pbd->m_vn);
		m_triangle_vertex.connect(&pbd->TriPoint);
		m_triangle_vertex_old.connect(&pbd->TriPointOld);
		m_triangle_index.connect(&pbd->Tri);
		pbd->setSmoothingLength(0.0085);
		

		//
		//pbd->setSmoothingLength(0.141421);

		this->setNumericalModel(pbd);
	}


	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::setInteractionDistance(Real d)
	{
		radius.setValue(d);
	}

	template<typename TDataType>
	SemiAnalyticalSFINode<TDataType>::~SemiAnalyticalSFINode()
	{
		
	}

	template<typename TDataType>
	bool SemiAnalyticalSFINode<TDataType>::initialize()
	{
		return true;
	}



	template<typename TDataType>
	bool SemiAnalyticalSFINode<TDataType>::resetStatus()
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

		if(total_num > 0)
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
			GArray<Coord>& allpoints = m_particle_position.getValue();
			GArray<Attribute>& allattrs = m_particle_attribute.getValue();


			std::vector<Attribute> attributeList;
			int sumTri = 0;
			int offset = 0;

			for (int i = 0; i < m_particleSystems.size(); i++)
			{
				if (m_particleSystems[i]->currentPosition()->getElementCount() == 0)
				{
					continue;
				}
				GArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
				GArray<Coord>& vels = m_particleSystems[i]->currentVelocity()->getValue();

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

			CArray<Triangle> host_triangle;
			host_triangle.resize(num_triangle);

			printf("mesh tri size: %d %d %d\n", num_triangle,start_triangle ,start_vertex);

			Function1Pt::copy(host_triangle, triangle_index);
			for (int j = 0; j < num_triangle; j++)
			{
				host_triangle[j][0] += start_vertex;
				host_triangle[j][1] += start_vertex;
				host_triangle[j][2] += start_vertex;
			}
			//printf("%d\n", start_vertex);
			cudaMemcpy(m_triangle_index.getValue().begin() + start_triangle, host_triangle.begin(), num_triangle * sizeof(Triangle), cudaMemcpyHostToDevice);
		
			host_triangle.clear();

			start_vertex += num_vertex;
			start_triangle += num_triangle;
		}

		return true;
	}

	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::advance(Real dt)
	{
		//return;
		printf("inside advance\n");
		//return;
		std::vector<std::shared_ptr<ParticleSystem<TDataType>>> m_particleSystems = this->getParticleSystems();
		std::vector<std::shared_ptr<TriangularSurfaceMeshNode<TDataType>>> m_surfaces = this->getTriangularSurfaceMeshNodes();
		bool emitter = false;
		if(emitter)
		{

			int total_num = 0;

			std::vector<int> ids;
			std::vector<int> fxd;
			std::vector<Real> mass;
			for (int i = 0; i < m_particleSystems.size(); i++)
			{
				//printf("%s\n", m_particleSystems[i]->getName().c_str());
				if (!(m_particleSystems[i]->getName().c_str()[0] == 'f')) // solid
				{
					continue;
				}
				if (m_particleSystems[i]->currentPosition()->getElementCount() == 0)
				{
				//	printf("out0\n");
					continue;
				}
				auto points = m_particleSystems[i]->currentPosition()->getValue();
				total_num += points.size();

				Real m = m_particleSystems[i]->getMass() * 10.0;// / points.size() * 100000.0;

				for (int j = 0; j < points.size(); j++)
				{
					ids.push_back(i);
					mass.push_back(m);
				}
			}
			printf("%d\n", total_num);

			if (total_num > 0)
				m_objId.resize(total_num);
			m_particle_velocity.setElementCount(total_num);
			m_particle_mass.setElementCount(total_num);
			m_particle_position.setElementCount(total_num);
			m_particle_force_density.setElementCount(total_num);
			//ParticleId.setElementCount(total_num);

			if (total_num > 0)
			{
				//posBuf.resize(total_num);
				//weights.resize(total_num);
				//init_pos.resize(total_num);
				//VelBuf.resize(total_num);
			


				//Function1Pt::copy(m_objId, ids);
				//Function1Pt::copy(ParticleId.getValue(), ids);
				Function1Pt::copy(m_particle_mass.getValue(), mass);

			}
			

			fxd.clear();
			ids.clear();
			mass.clear();

			m_particle_attribute.setElementCount(total_num);
			
			if(total_num > 0)
			{ 
				int start = 0;
				GArray<Coord>& allpoints = m_particle_position.getValue();
				GArray<Attribute>& allattrs = m_particle_attribute.getValue();


				std::vector<Attribute> attributeList;
				int offset = 0;

				for (int i = 0; i < m_particleSystems.size(); i++)
				{
					if (m_particleSystems[i]->currentPosition()->getElementCount() == 0)
					{
						continue;
					}
					GArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
					GArray<Coord>& vels = m_particleSystems[i]->currentVelocity()->getValue();

					int num = points.size();

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
			
			
		}


		auto nModel = this->getNumericalModel();
		if(m_particle_position.getElementCount() > 0)
		{
			printf("+++++++++++++++++++++++++++++++++++++++++++++++++++ %d %d\n", m_particle_position.getElementCount(), m_particle_velocity.getElementCount());
			nModel->step(this->getDt());
		}
		
		if(m_particle_position.getElementCount() > 0)
		{ 
			int start = 0;
			GArray<Coord>& allvels = m_particle_velocity.getValue();
			GArray<Coord>& allposs = m_particle_position.getValue();

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
				GArray<Coord>& points = m_particleSystems[i]->currentPosition()->getValue();
				GArray<Coord>& vels = m_particleSystems[i]->currentVelocity()->getValue();
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