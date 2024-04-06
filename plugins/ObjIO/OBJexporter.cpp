#pragma once
#include "OBJexporter.h"



namespace dyno
{
	IMPLEMENT_TCLASS(OBJExporter, TDataType)

		template<typename TDataType>
	OBJExporter<TDataType>::OBJExporter()
	{
		
		this->varEndFrame()->setRange(0,999999999);
		this->varFrameStep()->setRange(0, 999999999);
		this->varStartFrame()->setRange(0, 999999999);

		this->inTriangleSet()->tagOptional(true);
		this->inPolygonSet()->tagOptional(true);
	}

	template<typename TDataType>
	void OBJExporter<TDataType>::updateStates()
	{

		auto polySet = this->inPolygonSet()->getDataPtr();
		if (!this->inPolygonSet()->isEmpty()) 
		{
			this->outputPolygonSet(polySet);
		}

		auto triangleset = this->inTriangleSet()->getDataPtr();
		if (!this->inTriangleSet()->isEmpty()) 
		{
			this->outputTriangleMesh(triangleset);
		}
	}


	//reset
	template<typename TDataType>
	void OBJExporter<TDataType>::resetStates()
	{
	}

	template<typename TDataType>
	void OBJExporter<TDataType>::outputTriangleMesh(std::shared_ptr<TriangleSet<TDataType>> triangleset)
	{
		auto frame_step = this->varFrameStep()->getData();
		auto current_frame = this->stateFrameNumber()->getData();
		auto mode = this->varOutputType()->getValue();

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

		output << "# exported by PeriDyno (www.peridyno.com)" << std::endl;
		output << "# "<< host_vertices.size() << " points" << std::endl;
		if(mode == OutputType::Mesh)
			output << "# " << host_triangles.size() << " triangles" << std::endl;
		output << "g" << std::endl;

		for (uint i = 0; i < host_vertices.size(); ++i) {
			output << "v " << host_vertices[i][0] << " " << host_vertices[i][1] << " " << host_vertices[i][2] << std::endl;
		}
		if (mode == OutputType::Mesh) 
		{
			for (uint i = 0; i < host_triangles.size(); ++i) {
				output << "f " << host_triangles[i][0] + 1 << " " << host_triangles[i][1] + 1 << " " << host_triangles[i][2] + 1 << std::endl;
			}
		}

		output.close();


		host_vertices.clear();
		host_triangles.clear();


		return;
	}


	template<typename TDataType>
	void OBJExporter<TDataType>::outputPolygonSet(std::shared_ptr<PolygonSet<TDataType>> polygonSet)
	{		
		auto mode = this->varOutputType()->getData();

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

		auto polySet = this->inPolygonSet()->getDataPtr();

		auto d_points = polySet->getPoints();
		CArray<Coord> c_points;

		if (d_points.size())
			c_points.assign(d_points);
		else
			return;

		auto d_polygons = polySet->polygonIndices();

		CArrayList<uint> c_polygons;
		if(d_polygons.size())
			c_polygons.assign(d_polygons);
		else
			return;


		std::stringstream ss; ss << out_number;
		std::string filename;


		if (this->inTriangleSet()->isEmpty()) 
			filename = this->varOutputPath()->getData() + ss.str() + this->file_postfix;// output PolygonSet
		else
			filename = this->varOutputPath()->getData() + "Poly" + ss.str() + this->file_postfix;// output TriangleSet and PolygonSet

		std::ofstream output(filename.c_str(), std::ios::out);

		std::cout << filename << std::endl;

		if (!output.is_open()) {
			printf("------Polygon Writer: open file failed \n");
			return;
		}
		std::cout << "------Polygon Writer Action!------ " << std::endl;

		output << "# exported by PeriDyno (www.peridyno.com)" << std::endl;
		output << "# " << c_points.size() << " points" << std::endl;
		if (mode == OutputType::Mesh)
			output << "# " << c_polygons.size() << " primitives" << std::endl;
		output << "g" << std::endl;

		for (uint i = 0; i < c_points.size(); ++i) {
			output << "v " << c_points[i][0] << " " << c_points[i][1] << " " << c_points[i][2] << std::endl;
		}

		if (mode == OutputType::Mesh) 
		{
			for (size_t i = 0; i < c_polygons.size(); i++)
			{
				output << "f ";

				for (size_t j = 0; j < c_polygons[i].size(); j++)
				{
					output << c_polygons[i][j] + 1 << " ";
				}
				output << std::endl;
			}
		}

		output.close();

		c_points.clear();
		c_polygons.clear();
	}

	DEFINE_CLASS(OBJExporter)
}