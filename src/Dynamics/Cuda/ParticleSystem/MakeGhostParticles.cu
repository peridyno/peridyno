#include "MakeGhostParticles.h"

namespace dyno
{

	template<typename TDataType>
	MakeGhostParticles<TDataType>::MakeGhostParticles()
		: GhostParticles<TDataType>()
	{
		this->stateNormal()->promoteInput();
	}

	template<typename TDataType>
	MakeGhostParticles<TDataType>::~MakeGhostParticles()
	{
	}

	template<typename Coord>
	__global__ void MakeGhost_ReverseNomals(
		DArray<Coord> Normals,
		bool flag
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= Normals.size()) return;

		if (flag == true)
		{
			Normals[pId] = (-1.0) * Normals[pId];
		}

	}


	template<typename TDataType>
	void MakeGhostParticles<TDataType>::resetStates()
	{
		

		this->GhostParticles<TDataType>::resetStates();

		auto& inTopo = this->inPoints()->getData();

		auto pts = std::make_shared<PointSet<TDataType>>();
		pts->copyFrom(inTopo);

		int num = pts->getPoints().size();

		if (this->stateNormal()->size() != num)
		{
			std::cout << "Ghost particle normals error: " << num << ", " << this->stateNormal()->size() << std::endl;
		}

		this->statePointSet()->setDataPtr(pts);
		this->statePosition()->assign(pts->getPoints());

		this->stateVelocity()->allocate();
		this->stateVelocity()->resize(num);
		this->stateVelocity()->reset();

		std::vector<Attribute> host_attribute;
		Attribute attri;
		attri.setRigid();
		attri.setFixed();
		for (int i = 0; i < num; i++)
		{
			host_attribute.push_back(attri);
		}
		this->stateAttribute()->resize(num);
		this->stateAttribute()->assign(host_attribute);

		cuExecute(num,
			MakeGhost_ReverseNomals,
			this->stateNormal()->getData(),
			this->varReverseNormal()->getValue()
			);
	}

	DEFINE_CLASS(MakeGhostParticles);
}