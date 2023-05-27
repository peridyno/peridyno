#pragma once
#include "PointSet.h"

namespace dyno
{
	class EdgeSet : public PointSet
	{
	public:
		EdgeSet();
		~EdgeSet() override;

		DArray<Edge>& getEdges() { return mEdgeIndex; }

	protected:
		/**
		 * Override updateEdges to update edges in a special way
		 */
		virtual void updateEdges() {};

		void updateTopology() override;

	public:
		DArray<Edge> mEdgeIndex;
	};
}

