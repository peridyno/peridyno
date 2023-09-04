#pragma once
#include "OBJexporter.h"


namespace dyno
{
	IMPLEMENT_TCLASS(OBJExporter, TDataType)

		template<typename TDataType>
	OBJExporter<TDataType>::OBJExporter()
	{



	}

	template<typename TDataType>
	void OBJExporter<TDataType>::updateStates()
	{
		auto triangleset = this->inTriangleSet()->getDataPtr();
		auto mode = this->varOutputType()->getValue();
		
		if (mode == OutputType::TriangleMesh)
			outputSurfaceMesh(triangleset);
		else if(mode ==OutputType::PointCloud)
		{
			auto ptSet = TypeInfo::cast<PointSet<TDataType>>(this->inTriangleSet()->getDataPtr());
			outputPointCloud(ptSet);
		}




	}


	//reset
	template<typename TDataType>
	void OBJExporter<TDataType>::resetStates()
	{



	}

	template<typename TDataType>
	void OBJExporter<TDataType>::outputSurfaceMesh(std::shared_ptr<TriangleSet<TDataType>> triangleset)
	{
		auto frame_step = this->varFrameStep()->getData();
		auto current_frame = this->stateFrameNumber()->getData();

		if (this->stateFrameNumber()->getData() % this->varFrameStep()->getData() != 0)
		{
			printf("Skip Frame !\n");
			return;
		}

		unsigned out_number;
		if (frame_step > 1)
		{
			out_number = current_frame / frame_step;
		}
		else
		{
			out_number = this->stateFrameNumber()->getData();
		}

		std::stringstream ss; ss << out_number;
		std::string filename = this->varOutputPath()->getData() + ss.str() + this->file_postfix;// 
		std::ofstream output(filename.c_str(), std::ios::out);

		std::cout << filename << std::endl;

		if (!output.is_open()) {
			printf("------Triangle Mesh Writer: open file failed \n");
			return;
		}

		std::cout << "------Triangle Mesh Writer Action!------ " << std::endl;


		CArray<Coord> host_vertices;
		CArray<TopologyModule::Triangle> host_triangles;


		if (triangleset->getPoints().size())
		{
			host_vertices.assign(triangleset->getPoints());
		}
		if (triangleset->getTriangles().size())
		{
			host_triangles.assign(triangleset->getTriangles());
		}


		for (uint i = 0; i < host_vertices.size(); ++i) {
			output << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
		}
		for (uint i = 0; i < host_triangles.size(); ++i) {
			output << "f " << host_triangles[i][0] + 1 << " " << host_triangles[i][1] + 1 << " " << host_triangles[i][2] + 1 << std::endl;
		}
		output.close();


		host_vertices.clear();
		host_triangles.clear();


		return;
	}

	template<typename TDataType>
	void OBJExporter<TDataType>::outputPointCloud(std::shared_ptr<PointSet<TDataType>> pointset)
	{

		auto frame_step = this->varFrameStep()->getData();
		auto current_frame = this->stateFrameNumber()->getData();

		if (current_frame % frame_step != 0)
		{
			printf("Skip Frame !\n");
			return;
		}

		unsigned out_number;
		if (frame_step > 1)
		{
			out_number = current_frame / frame_step;
		}
		else
		{
			out_number = this->stateFrameNumber()->getData();
		}


		std::stringstream ss; ss << out_number;
		std::string filename = this->varOutputPath()->getData() + ss.str() + this->file_postfix;// 
		std::ofstream output(filename.c_str(), std::ios::out);

		std::cout << filename << std::endl;

		if (!output.is_open()) {
			printf("------Triangle Mesh Writer: open file failed \n");
			return;
		}

		std::cout << "------Pointcloud Writer Action!------ " << std::endl;

		CArray<Coord> host_vertices;

		if (pointset->getPoints().size())
		{
			host_vertices.assign(pointset->getPoints());
		}

		for (uint i = 0; i < host_vertices.size(); ++i) {
			output << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
		}

		output.close();

		host_vertices.clear();


		return;
	}




	DEFINE_CLASS(OBJExporter)
}