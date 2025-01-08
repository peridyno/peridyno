#include "PolyExtrude.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include <sstream> 


namespace dyno
{
	template<typename TDataType>
	PolyExtrude<TDataType>::PolyExtrude()
	{
		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		auto glModule2 = std::make_shared<GLPointVisualModule>();
		glModule2->setColor(Color(1, 0.1, 0.1));
		glModule2->varPointSize()->setValue(0.001);
		this->stateTriangleSet()->connect(glModule2->inPointSet());
		this->graphicsPipeline()->pushModule(glModule2);


		glModule3 = std::make_shared<GLWireframeVisualModule>();
		glModule3->setColor(Color(1, 1, 1));
		glModule3->varLineWidth()->setValue(0.1);
		glModule3->varForceUpdate()->setValue(true);

		this->stateTriangleSet()->connect(glModule3->inEdgeSet());
		this->graphicsPipeline()->pushModule(glModule3);

		this->varDistance()->setRange(-2,2);
		this->varDivisions()->setRange(1,50);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&PolyExtrude<TDataType>::varChanged, this));

		this->varDistance()->attach(callback);
		this->varDivisions()->attach(callback);
		this->varReverseNormal()->attach(callback);


	}

	template<typename TDataType>
	void PolyExtrude<TDataType>::resetStates() 
	{
		varChanged();
	}


	template<typename TDataType>
	void PolyExtrude<TDataType>::varChanged()
	{
		auto triangleSet = this->stateTriangleSet()->getDataPtr();
		
		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangles;
		extrude(vertices, triangles);

		triangleSet->setPoints(vertices);
		triangleSet->setTriangles(triangles);

		vertices.clear();
		triangles.clear();
		glModule3->update();
		triangleSet->update();
	}

	template<typename TDataType>
	void PolyExtrude<TDataType>::extrude(std::vector<Vec3f>& vertices, std::vector<TopologyModule::Triangle>& triangles)
	{
		auto triangleSet = this->stateTriangleSet()->getDataPtr();
		auto distance = this->varDistance()->getData();		
		auto divisions = this->varDivisions()->getData();
	
		std::string primString = this->varPrimitiveId()->getData();

		if (divisions == 0) { divisions = 1; }
		TriangleSet<TDataType>trilist;

		trilist.copyFrom(this->inTriangleSet()->getData());

		DArray<TopologyModule::Triangle> d_triangle = trilist.getTriangles();
		CArray<TopologyModule::Triangle> c_triangle;

		if (d_triangle.size()) 
		{
			c_triangle.assign(d_triangle);
		}
		else { return; }

		auto& sa = trilist.vertex2Edge();

		auto ss = trilist.getVertex2Triangles();

		DArray<Coord> d_point = trilist.getPoints();
		CArray<Coord> c_point;
		if (d_point.size()) 
		{
			c_point.assign(d_point);
		}

		DArray<TopologyModule::Edge> d_edges = trilist.getEdges();
		CArray<TopologyModule::Edge> c_edges;
		if (d_edges.size()) 
		{
			c_edges.assign(d_edges);
		}

		int n_point = this->inTriangleSet()->getDataPtr()->getPoints().size();
		int n_triangle = this->inTriangleSet()->getDataPtr()->getTriangles().size();
		int n_edges = this->inTriangleSet()->getDataPtr()->getEdges().size();

		CArray<uint> c_selected_primid;
		if (this->inPrimitiveId()->isEmpty()== false)
		{
			auto d_selected_primid = this->inPrimitiveId()->getData();
			c_selected_primid.assign(d_selected_primid);
		}



		std::vector<int> tempPrimArray;
		if (primString.size())
		{
			for (size_t i = 0; i < this->selectedPrimitiveID.size(); i++)
			{
				tempPrimArray.push_back(selectedPrimitiveID[i]);
			}
		}
		else 
		{
			for (size_t i = 0; i < c_selected_primid.size(); i++)
			{
				tempPrimArray.push_back(c_selected_primid[i]);
			}
			if (tempPrimArray.empty()) 
			{
				printf("selected empty  \n");
				return; 
			}
		}

		std::vector<int> selectedPtNum;

		for (int i = 0; i < tempPrimArray.size(); i++)
		{
			for (int j = 0; j < 3; j++)
			{
				for (int ptN = 0; ptN < selectedPtNum.size(); ptN++)
				{
					if (std::find(selectedPtNum.begin(), selectedPtNum.end(), c_triangle[i][ptN]) == selectedPtNum.end())
					{
					}
					else
					{
						selectedPtNum.push_back(c_triangle[i][ptN]);
					}
				}
			}
		}

		for (int i = 0; i < n_point; i++)
		{
			vertices.push_back(c_point[i]);
		}

		for (int i = 0; i < n_triangle; i++)
		{
			triangles.push_back(c_triangle[i]);
		}

		std::vector<int> tempid;
		std::map<int, int> ptid_count;
		std::map<int, int> ptid_allTri;

		for (int i = 0; i < tempPrimArray.size(); i++)
		{
			int n = tempPrimArray[i];
			int Tid;

			std::vector<int> idArray = { c_triangle[n][0] ,c_triangle[n][1] ,c_triangle[n][2] };

			for (int i = 0; i < idArray.size(); i++)
			{
				Tid = idArray[i]; 
				if (ptid_count.count(Tid))
				{
					ptid_count[Tid] = ptid_count.at(Tid) + 1;
				}
				else
				{
					ptid_count[Tid] = 1;
				}
			}
		}


		for (int i = 0; i < n_triangle; i++)
		{
			int Tid;
			std::vector<int> idArray = { c_triangle[i][0] ,c_triangle[i][1] ,c_triangle[i][2] };

			for (int k = 0; k < idArray.size(); k++)
			{
				Tid = idArray[k]; 
				if (ptid_count.count(Tid))
				{
					if (ptid_allTri.count(Tid))
					{
						ptid_allTri[Tid] = ptid_allTri.at(Tid) + 1;
					}
					else
					{
						ptid_allTri[Tid] = 1;
					}
				}
			}
		}



		std::vector<int> replace_id;
		std::vector<int> border_id;
		std::vector<int> select_ptid;

		for (auto it : ptid_count)
		{
			select_ptid.push_back(it.first);

			if (it.second == ptid_allTri.at(it.first))
			{
				replace_id.push_back(it.first);
			}
			else
			{
				border_id.push_back(it.first);
			}

		}

		 
		auto d_normal = this->inTriangleSet()->getDataPtr()->getVertexNormals();

		std::map<int, Coord> map_vertexID_tNormal;



		CArray<Coord> c_normal;
		c_normal.assign(d_normal);
		for (size_t i = 0; i < c_normal.size(); i++)
		{
			map_vertexID_tNormal[i] = c_normal[i];
		}

		std::vector<TopologyModule::Edge> normalDisplay;
		std::vector<Coord> normalCoord;
		float normalsize = 0.1f;
		for (size_t i = 0; i < c_point.size(); i++)
		{
			normalCoord.push_back(c_point[i]);
			Vec3f N = c_normal[i];
			Vec3f P = Coord(c_point[i][0], c_point[i][1], c_point[i][2]);
			Vec3f nP = P + N * normalsize;
			normalCoord.push_back(Coord(nP[0],nP[1],nP[2]));
			normalDisplay.push_back(TopologyModule::Edge(normalCoord.size(), normalCoord.size()-1));
		}


		std::map<point_layer, int> oriP_newP;

		for (size_t i = 0; i < border_id.size(); i++)
		{
			oriP_newP[point_layer(border_id[i], 0)] = border_id[i];
		}


		for (size_t k = 1; k < divisions; k++)
		{
			float tempdistance = float(k) /divisions * distance;

			for (int i = 0; i < border_id.size(); i++)
			{
				int temp = -1;

				Coord pt = vertices[border_id[i]];
				Coord N = map_vertexID_tNormal.at(border_id[i]);

				point_layer lp = point_layer(border_id[i], k);

				vertices.push_back(Coord(pt[0] + N[0] * tempdistance, pt[1] + N[1] * tempdistance, pt[2] + N[2] * tempdistance));

				oriP_newP[lp] = vertices.size() - 1;
				temp = vertices.size() - 1;
			}

		}

		for (int i = 0; i < select_ptid.size(); i++)
		{
			int temp = -1;

			Coord pt = vertices[select_ptid[i]];
			Coord N = map_vertexID_tNormal.at(select_ptid[i]);

			point_layer lp = point_layer(select_ptid[i], divisions);

			if (i < border_id.size())
			{
				vertices.push_back(Coord(pt[0] + N[0] * distance, pt[1] + N[1] * distance, pt[2] + N[2] * distance));

				oriP_newP[lp] = vertices.size() - 1;
				temp = vertices.size() - 1;
			}
			else
			{

				int tempi = i - border_id.size();
				vertices[replace_id[tempi]] = Coord(pt[0] + N[0] * distance, pt[1] + N[1] * distance, pt[2] + N[2] * distance);

				oriP_newP[lp] = replace_id[tempi];
				temp = replace_id[tempi];
			}


		}




		std::map<int, int> borderP_countInLine;
		std::vector<TopologyModule::Edge> borderLine;

		for (int i = 0; i < tempPrimArray.size(); i++)
		{
			bool one = std::count(border_id.begin(), border_id.end(), triangles[tempPrimArray[i]][0]);
			bool two = std::count(border_id.begin(), border_id.end(), triangles[tempPrimArray[i]][1]);
			bool three = std::count(border_id.begin(), border_id.end(), triangles[tempPrimArray[i]][2]);

			if (one && two)
			{
				borderLine.push_back(TopologyModule::Edge(triangles[tempPrimArray[i]][0], triangles[tempPrimArray[i]][1]));
			}
			if (two && three)
			{
				borderLine.push_back(TopologyModule::Edge(triangles[tempPrimArray[i]][1], triangles[tempPrimArray[i]][2]));
			}
			if (three && one)
			{
				borderLine.push_back(TopologyModule::Edge(triangles[tempPrimArray[i]][2], triangles[tempPrimArray[i]][0]));
			}
		}


		std::vector<TopologyModule::Edge> tempLine = borderLine;
		for (size_t i = 0; i < tempLine.size(); i++)
		{
			int k = tempLine.size() - i - 1;
			int row1_1 = tempLine[k][0];
			int row1_2 = tempLine[k][1];
			for (size_t j = 0; j < tempLine.size(); j++)
			{
				int row2_1 = tempLine[j][0];
				int row2_2 = tempLine[j][1];
				if (row1_1 == row2_2 && row1_2 == row2_1)
				{
					borderLine.erase(borderLine.begin() + k);
				}
			}
		}

		for (int i = 0; i < tempPrimArray.size(); i++)
		{
			triangles[tempPrimArray[i]][0] = oriP_newP.at(point_layer(triangles[tempPrimArray[i]][0],divisions));
			triangles[tempPrimArray[i]][1] = oriP_newP.at(point_layer(triangles[tempPrimArray[i]][1],divisions));
			triangles[tempPrimArray[i]][2] = oriP_newP.at(point_layer(triangles[tempPrimArray[i]][2],divisions));
		}

		for (size_t k = 0; k < divisions; k++)
		{
			for (size_t i = 0; i < borderLine.size(); i++)
			{
				point_layer pt_o1 = point_layer(borderLine[i][0],k);
				point_layer pt_o2 = point_layer(borderLine[i][1],k);

				point_layer pt_n1 = point_layer(borderLine[i][0],k + 1);
				point_layer pt_n2 = point_layer(borderLine[i][1],k + 1);

				triangles.push_back(TopologyModule::Triangle(oriP_newP.at(pt_o1), oriP_newP.at(pt_o2), oriP_newP.at(pt_n2)));
				triangles.push_back(TopologyModule::Triangle(oriP_newP.at(pt_o1), oriP_newP.at(pt_n2), oriP_newP.at(pt_n1)));
			}
		}


	

	}

	template<typename TDataType>
	void PolyExtrude<TDataType>::substrFromTwoString(std::string& first, std::string& Second, std::string& line, std::string& myStr, int& index)
	{
		if (index < int(line.size()))
		{
			size_t posStart = line.find(first, index);
			size_t posEnd = line.find(Second, posStart + 1);


			myStr = line.substr(posStart, posEnd - posStart);
			index = posEnd - 1;

			std::stringstream ss2(line);

		}
		else
		{
			return;
		}

	}


	DEFINE_CLASS(PolyExtrude);
}