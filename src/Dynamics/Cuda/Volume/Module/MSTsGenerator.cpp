#include "MSTsGenerator.h"
#include "MSTsGeneratorHelper.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"
#include <thrust/sort.h>
#include <ctime>
#include "Timer.h"
#include "STL/Stack.h"

namespace dyno
{
	IMPLEMENT_TCLASS(MSTsGenerator, TDataType)


	template<typename TDataType>
	void MSTsGenerator<TDataType>::compute()
	{
		MSTsGeneratorHelper<TDataType>::ConstructionFromScratch(
			this->inAGridSet()->getDataPtr(),
			this->inpMorton()->getData(),
			this->varLevelNum()->getData(),
			this->varOctreeType()->currentKey());

		AdaptiveGridGenerator<TDataType>::compute();
	}

	DEFINE_CLASS(MSTsGenerator);
}