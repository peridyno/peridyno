#include "initializeHeightField.h"

#include "Ocean.h"
#include "CapillaryWave.h"
#include "OceanPatch.h"

namespace dyno
{
	HeightFieldInitializer::HeightFieldInitializer()
	{
		
		TypeInfo::New<OceanPatch<DataType3f>>();
		//TypeInfo::New<Ocean<DataType3f>>();
		//TypeInfo::New<CapillaryWave<DataType3f>>();
		printf("222222222222222dfdffsdfsdfdsfds\n");

		
	}
}