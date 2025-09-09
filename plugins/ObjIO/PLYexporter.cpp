#pragma once
#include "PLYexporter.h"
#include "Topology/TetrahedronSet.h"
#include <fstream>
#include<iomanip>




namespace dyno
{
	IMPLEMENT_TCLASS(PlyExporter, TDataType)
	template <typename TDataType> class TriangleSet;

	template<typename TDataType>
	PlyExporter<TDataType>::PlyExporter()
	{	
		this->inVec3f()->tagOptional(true);
		this->inMatrix1()->tagOptional(true);
		this->inMatrix2()->tagOptional(true);
			
	}



	template<typename TDataType>
	void PlyExporter<TDataType>::updateStates()
	{
		if(m_frame > this->varEndFrame()->getValue()) return;
		m_frame ++;
		
		if (this->stateFrameNumber()->getData() % this->varFrameStep()->getData() != 0)
		{
			return;
		}

		TetrahedronSet<TDataType>tetlist;

		auto inTetSet = TypeInfo::cast<TetrahedronSet<DataType3f>>(this->inTopology()->getDataPtr());
		if (inTetSet != nullptr) 
		{
			tetlist.copyFrom(*inTetSet);
		}
		else 
		{
			auto ptSet = TypeInfo::cast<PointSet<DataType3f>>(this->inTopology()->getDataPtr());
			if (ptSet == nullptr) return;

			tetlist.setPoints(ptSet->getPoints());
		}


		auto d_tet = tetlist.tetrahedronIndices();
		CArray<TopologyModule::Tetrahedron> c_tet;
		if (d_tet.size()) 
		{
			c_tet.assign(d_tet);
		}

		DArray<Coord> d_point = tetlist.getPoints();
		CArray<Coord> c_point;
		c_point.assign(d_point);

		// DArray<Coord> d_color;
		CArray<Coord> c_color;
		// DArray<Matrix> d_strain;
		CArray<Matrix> c_strain;
		// DArray<Matrix> d_stress;
		CArray<Matrix> c_stress;

		if (!this->inVec3f()->isEmpty())
		{
			auto &d_color = this->inVec3f()->getData();
			c_color.assign(d_color);
		}
		if (!this->inMatrix1()->isEmpty())
		{
			auto & d_strain = this->inMatrix1()->getData();
			c_strain.assign(d_strain);
		}
		if (!this->inMatrix2()->isEmpty())
		{
			auto & d_stress = this->inMatrix2()->getData();
			c_stress.assign(d_stress);
		}

		int n_point = tetlist.vertex2Triangle().size();
		int n_triangle = tetlist.triangleIndices().size();
		int stw = 3;
		int num = 6;

		unsigned out_number;
		if (this->varReCount()->getData()) 
		{
			out_number = this->stateFrameNumber()->getData() / this->varFrameStep()->getData();
		}
		else 
		{
			out_number = this->stateFrameNumber()->getData();
		}

		std::stringstream ss; ss << out_number;
		std::string filename = this->varOutputPath()->getData() + ss.str() + this->file_postfix;// 
		std::ofstream out4(filename.c_str(), std::ios::out);

		out4 << "ply" << std::endl; 
		out4 << "format ascii 1.0" << std::endl;
		out4 << "comment created by Peridyno" << std::endl;
		out4 << "element vertex " << n_point <<  std::endl;

		out4 << "property float x" <<  std::endl;
		out4 << "property float y" <<  std::endl;
		out4 << "property float z" <<  std::endl;

		if (!this->inVec3f()->isEmpty())
		{
			out4 << "property uchar vec3f01" << std::endl;
			out4 << "property uchar vec3f02" << std::endl;
			out4 << "property uchar vec3f03" << std::endl;
		}

		if (!this->inMatrix1()->isEmpty())
		{
			out4 << "property float MatrixOne00" << std::endl;
			out4 << "property float MatrixOne01" << std::endl;
			out4 << "property float MatrixOne02" << std::endl;
			out4 << "property float MatrixOne10" << std::endl;
			out4 << "property float MatrixOne11" << std::endl;
			out4 << "property float MatrixOne12" << std::endl;
			out4 << "property float MatrixOne20" << std::endl;
			out4 << "property float MatrixOne21" << std::endl;
			out4 << "property float MatrixOne22" << std::endl;
		}

		if (!this->inMatrix2()->isEmpty())
		{
			out4 << "property float MatrixTwo00" << std::endl;
			out4 << "property float MatrixTwo01" << std::endl;
			out4 << "property float MatrixTwo02" << std::endl;
			out4 << "property float MatrixTwo10" << std::endl;
			out4 << "property float MatrixTwo11" << std::endl;
			out4 << "property float MatrixTwo12" << std::endl;
			out4 << "property float MatrixTwo20" << std::endl;
			out4 << "property float MatrixTwo21" << std::endl;
			out4 << "property float MatrixTwo22" << std::endl;
		}

		out4 << "element face " << n_triangle << std::endl;
		out4 << "property list uchar int vertex_indices" << std::endl;
		out4 << "end_header" << std::endl;


		
		// for (int i = 0; i < n_point; i++)
		for (int i = 0; i < n_point; i++)
		{
			//输出顶点坐标
			out4 << std::fixed << std::setw(stw) << std::setprecision(num) << c_point[i][0] << "  "
				<< std::fixed << std::setw(stw) << std::setprecision(num) << c_point[i][1] << "  "
				<< std::fixed << std::setw(stw) << std::setprecision(num) << c_point[i][2] << "  ";

			//输出颜色
			if (!this->inVec3f()->isEmpty()) {
				out4 << std::fixed << std::setw(stw) << std::setprecision(num) << c_color[i][0] << "  "
					<< std::fixed << std::setw(stw) << std::setprecision(num) << c_color[i][1] << "  "
					<< std::fixed << std::setw(stw) << std::setprecision(num) << c_color[i][2] << "  ";

			}

			//输出应变
			if (!this->inMatrix1()->isEmpty())
			{
				for (int j = 0; j < 3; j++) 
				{
					for (int k = 0; k < 3; k++)
					{
						out4 << std::fixed << std::setw(stw) << std::setprecision(num) << c_strain[i](j, k) << "  ";
					}
				}
			}
			//输出应力
			if (!this->inMatrix2()->isEmpty())
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						out4 << std::fixed << std::setw(stw) << std::setprecision(num) << c_stress[i](j, k) << "  ";
					}
				}
			}
			out4 << std::endl;
		}

		for (int i = 0; i < n_triangle; i++)
		{
			out4 << "3  " << std::fixed << std::setw(stw) << std::setprecision(num) << c_tet[i][2] << "  ";
			out4 << std::fixed << std::setw(stw) << std::setprecision(num) << c_tet[i][1] << "  ";
			out4 << std::fixed << std::setw(stw) << std::setprecision(num) << c_tet[i][0] 
				<<std::endl;
		}
		out4.close();


	}


	template<typename TDataType>
	void PlyExporter<TDataType>::resetStates()
	{

	}


	DEFINE_CLASS(PlyExporter)
}