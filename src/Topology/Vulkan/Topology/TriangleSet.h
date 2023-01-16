#pragma once
#include "EdgeSet.h"

namespace dyno
{
	class TriangleSet : public EdgeSet
	{
	public:
		TriangleSet();
		~TriangleSet() override;

	protected:
		void updateTopology() override;

		virtual void updateTriangles();

	public:
		DArray<Triangle> mTriangleIndex;
		DArray<uint32_t> mIndex;
	};
}

