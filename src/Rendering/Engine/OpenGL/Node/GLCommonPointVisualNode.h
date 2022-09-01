#pragma once
#include "Node.h"

#include "Topology/PointSet.h"

namespace dyno
{
	template<typename TDataType>
	class GLCommonPointVisualNode : public Node
	{
		DECLARE_TCLASS(GLCommonPointVisualNode, TDataType)
	public:
		typedef typename TDataType::Coord Coord;

		GLCommonPointVisualNode();
		~GLCommonPointVisualNode() override;

	public:
		std::string getNodeType() override;

	protected:
		void resetStates() override;

		DEF_INSTANCE_IN(PointSet<TDataType>, PointSet, "A set of points");

		DEF_VAR(Vec3f, Color, Vec3f(0,0,1.0f), "Color");
		DEF_VAR(Real, PointSize, 0.01f, "Point size");
	};

	IMPLEMENT_TCLASS(GLCommonPointVisualNode, TDataType)
};
