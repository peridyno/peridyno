#include "gmsh.h"
#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

namespace dyno
{

void Gmsh::loadFile(string filename)
{
    fstream filein(filename);
    if (!filein.is_open())
    {
        cout << "can't open Gmsh file:" << filename << endl;
        exit(0);
    }

	int ignored_lines = 0;
	bool version = false;
	bool node = false;
	//printf("YES\n");
	std::string line;
	while (!filein.eof()) {
		std::getline(filein, line);
		//std::cout << line << std::endl;
		//.obj files sometimes contain vertex normals indicated by "vn"
		if (line.substr(0, 1) != std::string("$")) {
			if (!version)
			{
				version = true;
			}
			else
			{
				std::stringstream data(line);
				int sum;
				data >> sum;
				//printf("sum = %d\n", sum);
				if (!node)
				{
					node = true;
					Real a, b, c;
					int idx;
					for (int i = 0; i < sum; i++)
					{
						//fscanf("%d%f%f%f", &idx, &a, &b, &c);
						
						std::getline(filein, line);
						std::stringstream data(line);
						data >> idx >> a >> b >> c;
						//std::cout << idx << ' ' << a << ' ' << b << ' ' << c << std::endl;
						m_points.push_back(Vec3f(a,b,c));
					}
					//printf("outside 1\n");
				}
				else
				{
					int idx1, idx2, idx3, idx4, idx5;
					int id1, id2, id3, id4;
					//printf("sum2 = %d\n", sum);

					for (int i = 0; i < sum; i++)
					{
						filein >> idx1 >> idx2 >> idx3 >> idx4 >> idx5;
						if (idx2 == 4)
						{
							filein >> id1 >> id2 >> id3 >> id4;
							//std::cout << id1 << ' ' << id2 << ' ' << id3 << ' ' << id4 << std::endl;
							/*if(
								(m_points[id1 - 1][0] < 0.85f || ((m_points[id1 - 1][2] > 0.7f * (m_points[id1 - 1][0] - 0.85f) || ((m_points[id1 - 1][2] < -0.7f * (m_points[id1 - 1][0] - 0.85f))))))
								&&						  																													
								(m_points[id2 - 1][0] < 0.85f || ((m_points[id2 - 1][2] > 0.7f * (m_points[id2 - 1][0] - 0.85f) || ((m_points[id2 - 1][2] < -0.7f * (m_points[id2 - 1][0] - 0.85f))))))
								&&						  																													
								(m_points[id3 - 1][0] < 0.85f || ((m_points[id3 - 1][2] > 0.7f * (m_points[id3 - 1][0] - 0.85f) || ((m_points[id3 - 1][2] < -0.7f * (m_points[id3 - 1][0] - 0.85f))))))
								&&						  																													
								(m_points[id4 - 1][0] < 0.85f || ((m_points[id4 - 1][2] > 0.7f * (m_points[id4 - 1][0] - 0.85f) || ((m_points[id4 - 1][2] < -0.7f * (m_points[id4 - 1][0] - 0.85f))))))
								)*/
							{ 
								m_tets.push_back(TopologyModule::Tetrahedron(id1 - 1, id2 - 1, id3 - 1, id4 - 1));
							}
						}
						else if (idx2 == 15)
						{
							filein >> id1;
						}
						else if (idx2 == 1)
						{
							filein >> id1 >> id2;
						}
						else 
						{
							filein >> id1 >> id2 >> id3;
						}
					}
					break;
				}
			}
			//std::stringstream data(line);
			

			//data >> point[0] >> point[1] >> point[2];
		}
		else {
			++ignored_lines;
		}
	}
	//printf("outside\n");
	filein.close();
}

} // namespace dyno
