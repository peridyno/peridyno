#include "SemiAnalyticalSFINode.h"
//#include "PBFM.h"
#include "SemiAnalyticalSurfaceTensionModel.h"
#include "SemiAnalyticalPositionBasedFluidModel.h"
#include "ParticleSystem/ParticleSystem.h"

namespace  dyno
{
	IMPLEMENT_TCLASS(SemiAnalyticalSFINode, TDataType)

	template<typename TDataType>
	SemiAnalyticalSFINode<TDataType>::SemiAnalyticalSFINode(std::string name)
		: Node(name)
	{
 		auto m_pbfModule = std::make_shared<SemiAnalyticalSurfaceTensionModel<DataType3f>>();
		
		this->stateTimeStep()->connect(m_pbfModule->inTimeStep());
		this->varSurfaceTension()->connect(m_pbfModule->varSurfaceTension());
		this->varAdhesionIntensity()->connect(m_pbfModule->varAdhesionIntensity());
		this->varRestDensity()->connect(m_pbfModule->varRestDensity());

		this->statePosition()->connect(m_pbfModule->inPosition());
		this->stateVelocity()->connect(m_pbfModule->inVelocity());
		this->stateForceDensity()->connect(m_pbfModule->inForceDensity());
		this->stateAttribute()->connect(m_pbfModule->inAttribute());

		this->stateTriangleIndex()->connect(m_pbfModule->inTriangleInd());
		this->stateTriangleVertex()->connect(m_pbfModule->inTriangleVer());

		this->animationPipeline()->pushModule(m_pbfModule);

		
		
	}

	template<typename TDataType>
	SemiAnalyticalSFINode<TDataType>::~SemiAnalyticalSFINode()
	{
	
	}

	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::preUpdateStates()
	{
		auto& particleSystems = this->getParticleSystems();

		int cur_num = this->statePosition()->size();

		if (particleSystems.size() > 0)
		{
			int new_num = 0;
			for (int i = 0; i < particleSystems.size(); i++)
			{
				new_num += particleSystems[i]->statePosition()->size();
			}

			if (new_num != cur_num)
			{
				this->statePosition()->resize(new_num);
				this->stateVelocity()->resize(new_num);
				this->stateForceDensity()->resize(new_num);
				this->stateAttribute()->resize(new_num);
			}
			if (this->statePosition()->size() <= 0)
				return;

			auto& new_pos = this->statePosition()->getData();
			auto& new_vel = this->stateVelocity()->getData();
			auto& new_force = this->stateForceDensity()->getData();

			//TODO: remove the follow code to improve performance
			CArray<Attribute> hostAttribute;
			hostAttribute.resize(statePosition()->size());
			for (int i = 0; i < statePosition()->size(); i++)
			{
				hostAttribute[i].setFluid();
				hostAttribute[i].setDynamic();
			}
			this->stateAttribute()->getDataPtr()->assign(hostAttribute);


			int start = 0;
			for (int i = 0; i < particleSystems.size(); i++)//update particle system
			{
				DArray<Coord>& points = particleSystems[i]->statePosition()->getData();
				DArray<Coord>& vels = particleSystems[i]->stateVelocity()->getData();
				DArray<Coord>& forces = particleSystems[i]->stateForce()->getData();
				int num = points.size();

				new_pos.assign(points, num, start);
				new_vel.assign(vels, num, start);
				new_force.assign(forces, num, start);

				start += num;
			}
		}
		
		auto& boundaryMeshes = this->getBoundaryMeshs();
		int start_vertex = 0;
		for (int i = 0; i < boundaryMeshes.size(); i++)
		{
			auto triSet = boundaryMeshes[i]->outTriangleSet()->getDataPtr();

			auto& vertex_position = triSet->getPoints();///!!!

			int num_vertex = vertex_position.size();

			cudaMemcpy(this->stateTriangleVertex()->getData().begin() + start_vertex, vertex_position.begin(), num_vertex * sizeof(Coord), cudaMemcpyDeviceToDevice);

			start_vertex += num_vertex;
		}
	}

	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::postUpdateStates()
	{
		auto& particleSystems = this->getParticleSystems();

		if (particleSystems.size() <= 0 || this->stateVelocity()->size() <= 0)
			return;

		auto& new_pos = this->statePosition()->getData();
		auto& new_vel = this->stateVelocity()->getData();
		auto& new_force = this->stateForceDensity()->getData();
		
		uint start = 0;
		for (int i = 0; i < particleSystems.size(); i++)//extend current particles
		{
			DArray<Coord>& points = particleSystems[i]->statePosition()->getData();
			DArray<Coord>& vels = particleSystems[i]->stateVelocity()->getData();
			DArray<Coord>& forces = particleSystems[i]->stateForce()->getData();
			int num = points.size();

			points.assign(new_pos, num, 0, start);
			vels.assign(new_vel, num, 0, start);
			forces.assign(new_force, num, 0, start);

			start += num;
		}
	}

	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::resetStates()
	{
		if (this->varFast()->getData() == true)
		{
			this->animationPipeline()->clear();
			auto pbd = std::make_shared<SemiAnalyticalPositionBasedFluidModel<DataType3f>>();
			pbd->varSmoothingLength()->setValue(0.0085);

			this->animationPipeline()->clear();
			this->stateTimeStep()->connect(pbd->inTimeStep());
			this->statePosition()->connect(pbd->inPosition());
			this->stateVelocity()->connect(pbd->inVelocity());
			this->stateForceDensity()->connect(pbd->inForce());
			this->stateTriangleVertex()->connect(pbd->inTriangleVertex());
			this->stateTriangleIndex()->connect(pbd->inTriangleIndex());
			this->animationPipeline()->pushModule(pbd);
		}

		auto& particleSystems = this->getParticleSystems();
		auto& boundaryMeshes = this->getBoundaryMeshs();

		//add particle system
		int total_num = 0;
		for (int i = 0; i < particleSystems.size(); i++)
		{
			total_num += particleSystems[i]->statePosition()->size();
		}
		//printf("total particles: %d\n", total_num);


		if (total_num > 0)
		{
			this->stateVelocity()->resize(total_num);
			this->statePosition()->resize(total_num);
			this->stateForceDensity()->resize(total_num);
			this->stateAttribute()->resize(total_num);


			int start = 0;
			DArray<Coord>& allpoints = this->statePosition()->getData();
			DArray<Attribute>& allattrs = this->stateAttribute()->getData();

			std::vector<Attribute> attributeList;
			int sumTri = 0;
			int offset = 0;

			for (int i = 0; i < particleSystems.size(); i++)
			{
				DArray<Coord>& points = particleSystems[i]->statePosition()->getData();
				DArray<Coord>& vels = particleSystems[i]->stateVelocity()->getData();

				int num = points.size();
				bool fixx = false;

				if (!(particleSystems[i]->getName().c_str()[0] == 'f')) // solid
				{
					for (int j = 0; j < num; j++)
					{
						Attribute a = Attribute();


						if (fixx)
						{
							a.setRigid();
							a.setFixed();
						}
						else
						{
							a.setRigid();
							a.setFixed();
						}

						attributeList.push_back(a);
					}
				}
				else//fluid
				{
					for (int j = 0; j < num; j++)
					{
						Attribute a = Attribute();
						a.setFluid();

						a.setDynamic();

						attributeList.push_back(a);
					}
				}


				cudaMemcpy(allpoints.begin() + start, points.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
				cudaMemcpy(this->stateVelocity()->getData().begin() + start, vels.begin(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);

				start += num;
			}
			this->stateAttribute()->getDataPtr()->assign(attributeList);
			attributeList.clear();
		}
		

		//add triangles
		int total_tri = 0;
		int total_tri_pos = 0;
		for (int i = 0; i < boundaryMeshes.size(); i++)
		{
			auto triSet = boundaryMeshes[i]->outTriangleSet()->getDataPtr();

			if (triSet != nullptr)
			{
				total_tri += triSet->getTriangles().size();
				total_tri_pos += triSet->getPoints().size();
			}
		}

		if (total_tri > 0 && total_tri_pos > 0)
		{
			this->stateTriangleIndex()->resize(total_tri);
			this->stateTriangleVertex()->resize(total_tri_pos);

			//printf("total triangles: %d\n", total_tri);

			int start_vertex = 0;
			int start_triangle = 0;

			for (int i = 0; i < boundaryMeshes.size(); i++)
			{
				auto triSet = boundaryMeshes[i]->outTriangleSet()->getDataPtr();

				auto& vertex_position = triSet->getPoints();///!!!

				int num_vertex = vertex_position.size();

				cudaMemcpy(this->stateTriangleVertex()->getData().begin() + start_vertex, vertex_position.begin(), num_vertex * sizeof(Coord), cudaMemcpyDeviceToDevice);

				auto& triangle_index = triSet->getTriangles();
				int num_triangle = triangle_index.size();

				CArray<Triangle> host_triangle;
				host_triangle.resize(num_triangle);
				host_triangle.assign(triangle_index);
				for (int j = 0; j < num_triangle; j++)
				{
					host_triangle[j][0] += start_vertex;
					host_triangle[j][1] += start_vertex;
					host_triangle[j][2] += start_vertex;
				}

				cudaMemcpy(this->stateTriangleIndex()->getData().begin() + start_triangle, host_triangle.begin(), num_triangle * sizeof(Triangle), cudaMemcpyHostToDevice);
				host_triangle.clear();

				start_vertex += num_vertex;
				start_triangle += num_triangle;
			}
		}
	}

	DEFINE_CLASS(SemiAnalyticalSFINode);
}