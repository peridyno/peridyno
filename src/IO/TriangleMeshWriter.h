/*
This Module is designed to output mesh file of TriangleSet;
the output file format: obj
*/

#pragma once
#include "Module/OutputModule.h"
#include "Module/TopologyModule.h"

#include "Topology/TriangleSet.h"

#include <string>
#include <memory>

namespace dyno
{
	template<typename TDataType>
	class TriangleMeshWriter : public OutputModule
	{
		DECLARE_TCLASS(TriangleMeshWriter, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		TriangleMeshWriter();
		virtual ~TriangleMeshWriter();

		void setNamePrefix(std::string prefix);
		void setOutputPath(std::string path);

		void setTriangleSetPtr(std::shared_ptr<TriangleSet<TDataType>> ptr_triangles) { this->ptr_TriangleSet = ptr_triangles;  this->updatePtr(); }
		bool updatePtr();
		bool updatePtr(TriangleSet<TDataType> triangle_set);

		bool outputSurfaceMesh();
		bool outputSurfaceMesh(TriangleSet<TDataType> triangleset );

	protected:
		void updateImpl() override;

	public:
		DEF_VAR_IN(unsigned, FrameNumber, "Input FrameNumber");
		DEF_INSTANCE_IN (TriangleSet<TDataType>, TriangleSet, "Input TriangleSet");

	protected:
		int time_idx = 0;
		int m_output_index = 0;
		int max_output_files = 10000;
		int idle_frame_num = 3;		//output one file of [num] frames
		int current_idle_frame = 0;
		std::string output_path = "G:/TEMP";
		std::string name_prefix = "cup";
		std::string file_postfix = ".obj";

		DArray<Triangle>* ptr_triangles;
		DArray<Coord>* ptr_vertices;
		std::shared_ptr<TriangleSet<TDataType>> ptr_TriangleSet = nullptr;
	};
}