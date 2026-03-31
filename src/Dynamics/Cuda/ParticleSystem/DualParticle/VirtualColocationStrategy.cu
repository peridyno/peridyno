#include "VirtualColocationStrategy.h"

#include "Node.h"
#include "ParticleSystem/Module/SummationDensity.h"

namespace dyno
{
	IMPLEMENT_TCLASS(VirtualColocationStrategy, TDataType)

	template<typename TDataType>
	VirtualColocationStrategy<TDataType>::VirtualColocationStrategy()
		: VirtualParticleGenerator<TDataType>()
	{

	}

	template<typename TDataType>
	VirtualColocationStrategy<TDataType>::~VirtualColocationStrategy()
	{

	}

	template <typename Coord>
	__global__ void VP2RP_RealCopytoVirtual(
		DArray<Coord> r_posArr,
		DArray<Coord> v_posArr
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= r_posArr.size()) return;

		v_posArr[pId] = r_posArr[pId];

	}

	template<typename TDataType>
	void VirtualColocationStrategy<TDataType>::constrain()
	{
		std::cout << "*DUAL-ISPH::ColocationStrategy(S.A.)" << std::endl;
	

		int num = this->inRPosition()->size();

	
		if (this->outVirtualParticles()->isEmpty())
		{
			this->outVirtualParticles()->allocate();
		}

		this->outVirtualParticles()->resize(num);

		cuExecute(num, VP2RP_RealCopytoVirtual,
			this->inRPosition()->getData(),
			this->outVirtualParticles()->getData()
		);
		//this->inVPosition()->connect(this->outVirtualParticles());
	}

	DEFINE_CLASS(VirtualColocationStrategy);

}