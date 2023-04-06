#pragma once
#include "Module/OutputModule.h"
#include "Module/TopologyModule.h"
#include "node.h"
#include "Topology/TriangleSet.h"
#include "TriangleMeshWriter.h"

#include <string>
#include <memory>

namespace dyno
{
	template <typename TDataType> class TriangleSet;

	template<typename TDataType>
	class OBJExporter : public Node
	{
		DECLARE_TCLASS(OBJExporter, TDataType)
	public:

		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		typedef typename Quat<Real> TQuat;

		DECLARE_ENUM(OutputType,
			TriangleMesh = 0,
			PointCloud = 1);

		OBJExporter();

		//void update() override;

	public:

		DEF_VAR(std::string, OutputPath, "D:/File_", "OutputPath");
		//DEF_VAR(std::string, Filename, "", "Filename");
		DEF_VAR(unsigned, StartFrame, 1, "StartFrame");
		DEF_VAR(unsigned, EndFrame, 500000, "EndFrame");
		DEF_VAR(unsigned, FrameStep, 1, "FrameStep");
		DEF_ENUM(OutputType, OutputType, TriangleMesh, "OutputType");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "TriangleSet")
		//DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSe, "TriangleSet");

		//DEF_NODE_PORTS(Node, Topology, "Topology");
		//DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");
		//DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		void resetStates() override;
		//void updateStates() override;

		void preUpdateStates() override;
		void setFrameStep();


	private:


		std::shared_ptr<TriangleMeshWriter<TDataType>> ExportModule = nullptr;
		std::shared_ptr<TriangleSet<TDataType>> ptr_TriangleSet = nullptr;


	};
}