#include "initializePeridynamics.h"

#include "ElasticityModule.h"
#include "ElastoplasticityModule.h"
#include "FractureModule.h"
#include "GranularModule.h"

namespace dyno 
{
	PeridynamicsInitializer::PeridynamicsInitializer()
	{
		TypeInfo::New<ElasticityModule<DataType3f>>();
		TypeInfo::New<ElastoplasticityModule<DataType3f>>();
		TypeInfo::New<FractureModule<DataType3f>>();
		TypeInfo::New<GranularModule<DataType3f>>();
	}
}