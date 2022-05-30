#include "initializePeridynamics.h"

#include "Module/ElasticityModule.h"
#include "Module/ElastoplasticityModule.h"
#include "Module/FractureModule.h"
#include "Module/GranularModule.h"

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