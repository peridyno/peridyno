#include "MSTsGenerator2D.h"
#include "Algorithm/Scan.h"
#include "Algorithm/Reduction.h"
#include <thrust/sort.h>
#include <ctime>
#include "Timer.h"
#include "STL/Stack.h"

#include "MSTsGeneratorHelper2D.h"

namespace dyno
{
	IMPLEMENT_TCLASS(MSTsGenerator2D, TDataType)

		template<typename TDataType>
	void MSTsGenerator2D<TDataType>::compute()
	{
		GTimer timer;
		timer.start();
		MSTsGeneratorHelper2D<TDataType>::ConstructionFromScratch2D(
			this->inAGridSet()->getDataPtr(),
			this->inpMorton()->getData(),
			this->varLevelNum()->getData(),
			this->varQuadType()->currentKey());
		timer.stop();
		printf("MSTGenerator2D:  %f \n", timer.getElapsedTime());
	}

	DEFINE_CLASS(MSTsGenerator2D);
}