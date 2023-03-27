#pragma once
#include "Module/TopologyMapping.h"

#include "Topology/DiscreteElements.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	class DiscreteElementsToTriangleSet : public TopologyMapping
	{
		DECLARE_CLASS(DiscreteElementsToTriangleSet);
	public:
		DiscreteElementsToTriangleSet();

	protected:
		bool apply() override;

	public:
		DEF_INSTANCE_IN(DiscreteElements, DiscreteElements, "");
		DEF_INSTANCE_OUT(TriangleSet, TriangleSet, "");

	private:
		TriangleSet mStandardSphere;
		TriangleSet mStandardCapsule;
	};
}