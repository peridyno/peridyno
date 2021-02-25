#include "ParticleFluidMesh.h"
#include "PositionBasedFluidModelMesh.h"

#include "Topology/PointSet.h"
//#include "PointRenderModule.h"
#include "Utility.h"
#include "DensitySummationMesh.h"

#include "Attribute.h"


#include "Mapping/PointSetToPointSet.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"


namespace dyno
{
	IMPLEMENT_CLASS_1(ParticleFluidMesh, TDataType)

	template<typename TDataType>
	ParticleFluidMesh<TDataType>::ParticleFluidMesh(std::string name)
		: ParticleSystem<TDataType>(name)
	{


		m_surfaceNode = this->template createChild<Node>("Mesh");


		auto pbf = this->template setNumericalModel<PositionBasedFluidModelMesh<TDataType>>("pbd");
		this->setNumericalModel(pbf);

		this->currentPosition()->connect(&pbf->m_position);
		this->currentVelocity()->connect(&pbf->m_velocity);
		this->currentForce()->connect(&pbf->m_forceDensity);

		this->m_attribute.connect(&pbf->m_attribute);
//		this->m_position2.connect(pbf->m_position2);
		this->m_normal2.connect(&pbf->m_normal);
	}

	template<typename TDataType>
	ParticleFluidMesh<TDataType>::~ParticleFluidMesh()
	{
		
	}


	template<typename TDataType>
	void ParticleFluidMesh<TDataType>::loadSurface(std::string filename)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->loadObjFile(filename);

	}
	template<typename TDataType>
	void ParticleFluidMesh<TDataType>::advance(Real dt)
	{
		auto nModel = this->getNumericalModel();
		nModel->step(this->getDt());
	}
}