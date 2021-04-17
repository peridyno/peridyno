#include "smesh.h"
#include <string.h>
#include <fstream>
#include <iostream>
using namespace std;

namespace dyno
{

void Smesh::loadFile(string filename)
{
    fstream filein(filename);
    if (!filein.is_open())
    {
        cout << "can't open smesh file:" << filename << endl;
        exit(0);
    }

    string part_str;
    filein >> part_str;
    if (part_str != "*VERTICES")
    {
        cout << "first non-empty line must be '*VERTICES'." << endl;
        exit(0);
    }
    int num_points = 0, point_dim = 0;
    filein >> num_points >> point_dim;
	int dummy;
	filein >> dummy >> dummy;
    m_points.resize(num_points, Vec3f(0.0f));
    for (int i = 0; i < num_points; ++i)
    {
        int vert_index;
        filein >> vert_index;
        for (int j = 0; j < point_dim; ++j)
        {
            filein >> m_points[i][j];
        }
    }

    filein >> part_str;
//     if (part_str != "*ELEMENTS")
//     {
//         cout << "after vertices, the first non-empty line must be '*ELEMENTS'." << endl;
//         return;
//     }

    while (!filein.eof())
    {
        string ele_type = "";
		int num_eles = 0, ele_dim = 0;
		filein >> ele_type >> num_eles >> ele_dim;
		int dummy;
		filein >> dummy;
        if (ele_type == "LINE")
        {
            m_edges.resize(num_eles);
            for (int i = 0; i < num_eles; ++i)
            {
                int ele_index;
                filein >> ele_index;
                for (int j = 0; j < ele_dim; ++j)
                {
                    filein >> m_edges[i][j];
					m_edges[i][j] = m_edges[i][j] - 1;
                }
            }
        }
        else if (ele_type == "TRIANGLE")
        {
            m_triangles.resize(num_eles);
            for (int i = 0; i < num_eles; ++i)
            {
                int ele_index;
                filein >> ele_index;
                for (int j = 0; j < ele_dim; ++j)
                {
                    filein >> m_triangles[i][j];
					m_triangles[i][j] = m_triangles[i][j] - 1;
                }
            }
        }
        else if (ele_type == "QUAD")
        {
            m_quads.resize(num_eles);
            for (int i = 0; i < num_eles; ++i)
            {
                int ele_index;
                filein >> ele_index;
                for (int j = 0; j < ele_dim; ++j)
                {
                    filein >> m_quads[i][j];
					m_quads[i][j] = m_quads[i][j] - 1;
                }
            }
        }
        else if (ele_type == "TET")
        {
            m_tets.resize(num_eles);
            for (int i = 0; i < num_eles; ++i)
            {
                int ele_index;
                filein >> ele_index;
                for (int j = 0; j < ele_dim; ++j)
                {
                    filein >> m_tets[i][j];
					m_tets[i][j] = m_tets[i][j] - 1;
                }
            }
        }
        else if (ele_type == "HEX")
        {
            m_hexs.resize(num_eles);
            for (int i = 0; i < num_eles; ++i)
            {
                int ele_index;
                filein >> ele_index;
                for (int j = 0; j < ele_dim; ++j)
                {
                    filein >> m_hexs[i][j];
					m_hexs[i][j] = m_hexs[i][j] - 1;
                }
            }
        }
        else
        {
            cout << "unrecognized element type:" << ele_type << endl;
        }
    }

	filein.close();
}

void Smesh::loadNodeFile(std::string filename)
{
	fstream filein(filename);
	if (!filein.is_open())
	{
		cout << "can't open node file:" << filename << endl;
		exit(0);
	}

	int num_points = 0, point_dim = 0;
	filein >> num_points >> point_dim;
	int dummy;
	filein >> dummy >> dummy;
	m_points.resize(num_points, Vec3f(0.0f));
	for (int i = 0; i < num_points; ++i)
	{
		int vert_index;
		filein >> vert_index;
		for (int j = 0; j < point_dim; ++j)
		{
			filein >> m_points[i][j];
		}
	}

	filein.close();
}

void Smesh::loadEdgeFile(std::string filename)
{
	fstream filein(filename);
	if (!filein.is_open())
	{
		cout << "can't open ele file:" << filename << endl;
		exit(0);
	}

	int num_of_edges = 0, edge_dim = 0;
	filein >> num_of_edges >> edge_dim;
	m_edges.resize(num_of_edges);

	int dummy;
	for (int i = 0; i < num_of_edges; ++i)
	{
		int vert_index;
		filein >> vert_index;
		
		filein >> m_edges[i][0] >> m_edges[i][1] >> dummy;
	}

	filein.close();
}

void Smesh::loadTriangleFile(std::string filename)
{
	fstream filein(filename);
	if (!filein.is_open())
	{
		cout << "can't open ele file:" << filename << endl;
		exit(0);
	}

	int dummy;

	int num_of_triangles = 0;
	filein >> num_of_triangles >> dummy;
	m_triangles.resize(num_of_triangles);

	for (int i = 0; i < num_of_triangles; ++i)
	{
		int vert_index;
		filein >> vert_index;

		filein >> m_triangles[i][0] >> m_triangles[i][1] >> m_triangles[i][2] >> dummy;
	}

	filein.close();
}

void Smesh::loadTetFile(std::string filename)
{
	fstream filein(filename);
	if (!filein.is_open())
	{
		cout << "can't open ele file:" << filename << endl;
		exit(0);
	}

	int ele_num = 0, ele_dim = 0;
	int dummy;
	filein >> ele_num >> ele_dim >> dummy;

	m_tets.resize(ele_num);
	for (int i = 0; i < ele_num; ++i)
	{
		int ele_index;
		filein >> ele_index;
		for (int j = 0; j < ele_dim; ++j)
		{
			filein >> m_tets[i][j];
		}
	}

	filein.close();
}

} // namespace dyno
