#pragma once
#include "EdgeSet.h"

namespace dyno
{
	class TriangleSet : public EdgeSet
	{
	public:
		TriangleSet();
		~TriangleSet() override;

		void setTriangles(std::vector<Triangle>& indices);
		void setTriangles(const DArray<Triangle>& indices);

		DArray<Triangle>& getTriangles() { return mTriangleIndex; }

		//TODO: fix the hack
		DArray<uint32_t>& getVulkanIndex() { return mIndex; }

	protected:
		void updateTopology() override;

		virtual void updateTriangles();

	public:
		DArray<Triangle> mTriangleIndex;
		DArray<uint32_t> mIndex;
	};
}

