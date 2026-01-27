#include "MSTsGeneratorLocalUpdate2D.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"
#include <thrust/sort.h>
#include <ctime>
#include "Timer.h"
#include "STL/Stack.h"

#include "MSTsGeneratorHelper2D.h"

namespace dyno
{
	IMPLEMENT_TCLASS(MSTsGeneratorLocalUpdate2D, TDataType)

	template<typename TDataType>
	void MSTsGeneratorLocalUpdate2D<TDataType>::compute()
	{
		if (this->inFrameNumber()->getData() <= 0)
		{
			MSTsGeneratorHelper2D<TDataType>::ConstructionFromScratch2D(
				this->inAGridSet()->getDataPtr(),
				this->inpMorton()->getData(),
				this->varLevelNum()->getData(),
				this->varQuadType()->currentKey());
		}
		else
		{
			printf("Dynamic Update Start \n");
			MSTsGeneratorHelper2D<TDataType>::DynamicUpdate2D(
				this->inAGridSet()->getDataPtr(),
				this->inpMorton()->getData(),
				this->inDecreaseMorton()->getData(),
				this->varLevelNum()->getData(),
				this->varQuadType()->currentKey());
		}
	}

	DEFINE_CLASS(MSTsGeneratorLocalUpdate2D);
}