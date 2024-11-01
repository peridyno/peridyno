#include "EarClipper.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include "EarClipper.h"

namespace dyno
{
	template<typename TDataType>
	EarClipper<TDataType>::EarClipper()
		: ModelEditing<TDataType>()
	{

	}

	template<typename TDataType>
	void EarClipper<TDataType>::resetStates()
	{
		varChanged();
	}

	template<typename TDataType>
	void EarClipper<TDataType>::varChanged()
	{
		auto d_coords = this->inPointSet()->getDataPtr()->getPoints();
		CArray<Coord> c_coords;
		c_coords.assign(d_coords);
		std::vector<DataType3f::Coord> vts;
		for (size_t i = 0; i < c_coords.size(); i++)
		{
			vts.push_back(c_coords[i]);
		}
		std::vector<TopologyModule::Triangle> outTriangles;
		polyClip(vts,outTriangles);


	}

	template<typename TDataType>
	void EarClipper<TDataType>::polyClip(std::vector<DataType3f::Coord> vts, std::vector<TopologyModule::Triangle>& outTriangles)
	{
		std::map<int, int> current_id_original_id;
		std::map<int, int> original_id_current_id;
		std::vector<int> earVertices;
		std::vector<DataType3f::Coord> tempVertices;
		Vec3f Standard_N;
		int MaxCount = vts.size() * 4;

		tempVertices.assign(vts.begin(), vts.end());

		for (int i = 0; i < vts.size(); i++)
		{
			current_id_original_id[i] = i;
			original_id_current_id[i] = i;
		}

		//printf(" vts : %d  \n",vts.size());

		int index = 0;

		while (1)
		{
			index++;
			bool convexPolygon = true;
			bool isear = false;

			//printf("\n");
			{
				Vec3f ConVertexXmin = vts[0];
				Vec3f ConVertexXmax = vts[0];
				Vec3f ConVertexZmin = vts[0];
				std::vector<int> dir = { 0,0,0 };
				for (int i = 0; i < vts.size(); i++)
				{
					if (vts[i][0] < ConVertexXmin[0]) { ConVertexXmin = vts[i]; dir[0] = i; }
					if (vts[i][0] > ConVertexXmax[0]) { ConVertexXmax = vts[i]; dir[1] = i; }
					if (vts[i][2] < ConVertexZmin[2]) { ConVertexZmin = vts[i]; dir[2] = i; }
				}
				std::sort(dir.begin(), dir.end());
				Vec3f v1 = vts[dir[1]] - vts[dir[0]];
				Vec3f v2 = vts[dir[2]] - vts[dir[1]];

				Standard_N = v2.cross(v1).normalize();
				/*printf("stand N = %g  %g  %g\n", v1[0],v1[1],v1[2]);
				printf("stand N = %g  %g  %g\n", v2[0],v2[1],v2[2]);
				printf("stand N = %g  %g  %g\n", Standard_N[0],Standard_N[1],Standard_N[2]);*/
			}

			//printf("find ear\n");
			for (int i = 0; i <= vts.size(); i++)
			{
				int f = i;
				int s = i + 1;
				int t = i + 2;

				if (i == vts.size() - 2) { t = 0; }

				else if (i == vts.size() - 1) { t = 1; s = 0; }

				else if (i == vts.size()) { f = 0; s = 1; t = 2; }

				//else if (i == vts.size()) { f = vts.size() - 1; s = 0; t = 1; }

				//printf("f : %d , s : %d , t : %d\n",f,s,t);
				//判断方向
				Vec3f v1 = vts[s] - vts[f];
				Vec3f v2 = vts[t] - vts[s];

				Vec3f N0 = v2.cross(v1).normalize();

				auto lamd = [=](Vec3f N, Vec3f SN)->bool
				{
					if (SN[1] > 0)
					{
						//printf("凹点\n"); 
						return N[1] < 0;
					}
					else {
						//printf("凸点\n"); 
						return N[1] > 0;
					}
				};

				//printf("N0 , %g   %g   %g   \n", N0[0], N0[1], N0[2]);

				if (lamd(N0, Standard_N))//
				{
					convexPolygon = false;
					isear = true;
				}

				else
				{
					if (isear == true)
					{
						bool clear = true;
						for (int j = 0; j < vts.size(); j++)
						{
							Vec3f p = vts[j];

							Vec3f v01 = vts[s] - vts[f];
							Vec3f v02 = vts[f] - vts[t];
							Vec3f v03 = vts[t] - vts[f];

							Vec3f vx01 = vts[f] - p;
							Vec3f vx02 = vts[t] - p;
							Vec3f vx03 = vts[s] - p;

							Vec3f N01 = v01.cross(vx01).normalize();
							Vec3f N02 = v02.cross(vx02).normalize();
							Vec3f N03 = v03.cross(vx03).normalize();

							//std::cout << std::endl; 
							//std::cout << "currentP: " << j << std::endl;
							//std::cout << "s: " << s << std::endl;

							//std::cout << "N01： " << N01[0] << "  " << N01[1] << "  " << N01[2] << std::endl;
							//std::cout << "N02： " << N02[0] << "  " << N02[1] << "  " << N02[2] << std::endl;
							//std::cout << "N03： " << N03[0] << "  " << N03[1] << "  " << N03[2] << std::endl;

							if (N01[1] < 0 && N02[1] < 0 && N03[1] < 0)
							{
								clear = false;
							}
							//std::cout << "clear: " << int(clear) << std::endl;
							//std::cout << std::endl;
						}

						if (clear)
						{
							earVertices.push_back(s);
						}

						//std::cout << "add： " << s << std::endl;
						isear = false;

					}

				}

			}

			std::sort(earVertices.begin(), earVertices.end());

			while (earVertices.size())
			{
				int i = earVertices[earVertices.size() - 1];
				int f = i - 1;
				int s = i;
				int t = i + 1;
				if (f < 0) { f = vts.size() - 1; }
				if (t > vts.size() - 1) { t = 0; }

				outTriangles.push_back(TopologyModule::Triangle(current_id_original_id.find(f)->second, current_id_original_id.find(s)->second, current_id_original_id.find(t)->second));


				//for (auto it : current_id_original_id)
				//{
				//	printf("%d : %d \n", it.first, it.second);

				//}

				//printf("删除点序号：%d \n", s);

				//printf("当前处于： %d \n", original_id_current_id.find(s)->second);

				vts.erase(vts.begin() + current_id_original_id.find(s)->first);

				for (int k = 0; k < current_id_original_id.size() - 1; k++)
				{
					if (current_id_original_id.find(k)->second >= current_id_original_id.find(s)->second)//当前顶点序号大于等于删除的序号，把下边点序号拿上来
					{
						auto it = current_id_original_id.find(k);
						auto it2 = current_id_original_id.find(k + 1);
						it->second = it2->second;
					}
				}


				//printf("查表\n");
				//printf("删除耳朵序号： %d\n", earVertices.size() - 1);
				earVertices.erase(earVertices.begin() + earVertices.size() - 1);
				//printf("删除完成");

			}


			//printf(" vts : %d  \n", vts.size());

			if (convexPolygon)
			{
				//printf("convexPolygon = true");

				//凸多边形分解
				for (int i = 1; i < vts.size() - 1; i++)
				{
					outTriangles.push_back(TopologyModule::Triangle(current_id_original_id.find(0)->second, current_id_original_id.find(i)->second, current_id_original_id.find(i + 1)->second));
				}

				break;
			}
			else if (vts.size() <= 3)
			{
				//printf("vts.size() <= 3 ");
				outTriangles.push_back(TopologyModule::Triangle(current_id_original_id.find(0)->second, current_id_original_id.find(1)->second, current_id_original_id.find(2)->second));

				break;
			}
			else { convexPolygon = true; }
			//printf(" Triangle construction completed \n"); 

			if (index >= MaxCount)
			{
				printf("Forced break\n");
				break;
			}
			earVertices.clear();
			//	printf("earVertices.clear();\n");

		}

	}

	DEFINE_CLASS(EarClipper);
}