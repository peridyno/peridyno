#include "SphereModel.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	SphereModel<TDataType>::SphereModel()
		: ParametricModel<TDataType>()
	{
		this->varRadius()->setRange(0.001f, 10.0f);

		this->varTheta()->setRange(0.001f, 1.5f);

		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		
		this->varRow()->setRange(8, 10000);
		this->varColumns()->setRange(8, 10000);


		glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Vec3f(0.8, 0.52, 0.25));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);


		//auto glPointModule = std::make_shared<GLPointVisualModule>();
		//glPointModule->setColor(Vec3f(0.8, 0, 0));
		//glPointModule->setVisible(true);
		//glPointModule->varPointSize()->setValue(0.01);
		//this->stateTriangleSet()->connect(glPointModule->inPointSet());
		//this->graphicsPipeline()->pushModule(glPointModule);


		auto wireframe = std::make_shared<GLWireframeVisualModule>();
		this->stateTriangleSet()->connect(wireframe->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireframe);

		this->stateTriangleSet()->promoteOuput();
	}

	template<typename TDataType>
	void SphereModel<TDataType>::resetStates()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		auto radius = this->varRadius()->getData();
		auto theta = this->varTheta()->getData();

		auto row = this->varRow()->getData();
		auto columns = this->varColumns()->getData();

		auto mode = this->varSphereMode()->getData();


		//radius *= (sqrt(scale[0] * scale[0] + scale[1] * scale[1] + scale[2] * scale[2]));

		TSphere3D<Real> ball;
		ball.center = center;
		ball.radius = radius;
		this->outSphere()->setValue(ball);


		auto triangleSet = this->stateTriangleSet()->getDataPtr();

		std::vector<Coord> vertices;
		std::vector<TopologyModule::Triangle> triangle;

		if (mode == SphereMode::Theta) 
		{


			Real PI = 3.1415926535;



			Coord point(0.0f);

			for (float d_alpha = -PI/2 + theta/2; d_alpha < PI/2; d_alpha += theta )
			{
				point[1] = center[1] + radius * sin(d_alpha);

				for (float d_beta = 0.0f; d_beta < 2 * PI; d_beta += theta )
				{
					point[0] = center[0] + (cos(d_alpha) * radius) * sin(d_beta);
					point[2] = center[2] + (cos(d_alpha) * radius) * cos(d_beta);
					vertices.push_back(point);

				}

			}

			vertices.push_back(Coord(center[0], -radius + center[1], center[2]));
			vertices.push_back(Coord(center[0], radius + center[1], center[2]));


			int face_id = 0;
			for (float d_alpha = -PI / 2 + theta/2; d_alpha < PI / 2; d_alpha += theta)
			{


				for (float d_beta = 0.0f; d_beta < 2 * PI; d_beta += theta)
				{
					if ((d_beta + theta - 2 * PI < EPSILON)&&(d_alpha + theta < PI / 2))
					{
						int cd = 2 * PI / theta;
						triangle.push_back(TopologyModule::Triangle(face_id, face_id+1, face_id + cd + 1));
					}
					else if((d_beta + theta - 2 * PI >= EPSILON) && (d_alpha + theta < PI / 2))
					{
						int cd = 2 * PI / theta;
						triangle.push_back(TopologyModule::Triangle(face_id, face_id - cd , face_id + cd + 1));
					}
					else if ((d_beta + theta - 2 * PI < EPSILON) && (d_alpha + theta >= PI / 2))
					{
						triangle.push_back(TopologyModule::Triangle(face_id, face_id + 1, vertices.size() - 1));
					}
					else if ((d_beta + theta - 2 * PI >= EPSILON) && (d_alpha + theta >= PI / 2))
					{
						int cd = 2 * PI / theta;
						triangle.push_back(TopologyModule::Triangle(face_id, face_id - cd, vertices.size() - 1));
					}



					if ((d_beta + theta - 2 * PI < EPSILON) && (d_alpha - theta + PI / 2 > EPSILON ))
					{
						int cd = 2 * PI / theta;
						triangle.push_back(TopologyModule::Triangle(face_id, face_id - cd, face_id + 1));
					}
					else if ((d_beta + theta - 2 * PI > EPSILON) && (d_alpha - theta + PI / 2 > EPSILON))
					{
						int cd = 2 * PI / theta;
						triangle.push_back(TopologyModule::Triangle(face_id, face_id - 2 * cd - 1, face_id - cd));
					}
					else if ((d_beta + theta - 2 * PI < EPSILON) && (d_alpha - theta + PI / 2 < EPSILON))
					{
						triangle.push_back(TopologyModule::Triangle(face_id, vertices.size() - 2, face_id + 1));
					}
					else if ((d_beta + theta - 2 * PI >= EPSILON) && (d_alpha - theta + PI / 2 < EPSILON))
					{
						int cd = 2 * PI / theta;
						triangle.push_back(TopologyModule::Triangle(face_id, vertices.size() - 2, face_id - cd));
					}

					face_id++;
				}
			}
		}


//RowAndColumns
		else if (mode == SphereMode::RowAndColumns)
		{
			vertices.clear();
			Real PI = 3.1415926535;

			//std::cout.setf(std::ios::fixed);
			//std::cout.precision(2);
			uint counter = 0;
			Coord Location;
			float angle = PI / 180 * 360 / columns;
			float temp_angle = angle;
			float x, y, z;
			float Theta_y = PI / 180 * 360 / (2 * (row - 1));
			float temp_Theta = Theta_y;

			float ThetaValue = 360 / (2 * (row - 1));
			float temp_ThetaValue;
			float test = PI / 180 * 30;

			int forNum;
			if (row % 2 == 0) { forNum = (row - 2) / 2; }
			else { forNum = (row - 1) / 2; }

			for (int i = 0; i < forNum; i++)
			{
				temp_Theta = i * Theta_y;
				temp_ThetaValue = i * ThetaValue;

				if (row % 2 == 0) 
				{
					temp_Theta = i * Theta_y + Theta_y / 2;
					temp_ThetaValue = i * ThetaValue + ThetaValue / 2;
					if (temp_ThetaValue >= 0 && temp_ThetaValue <= 90)
					{
						y = sin(temp_Theta) * radius;
					}
					else if (temp_ThetaValue >= 90 && temp_ThetaValue <= 180)
					{

					}
					else
					{
						break;
					}


				}
				else
				{
					temp_Theta = i * Theta_y;
					temp_ThetaValue = i * ThetaValue;
					if (temp_ThetaValue >= 0 && temp_ThetaValue <= 90)
					{
						y = sin(temp_Theta) * radius;
					}
					else if (temp_ThetaValue >= 90 && temp_ThetaValue <= 180)
					{
					}
					else
					{
						break;
					}
				}


				for (int k = 0; k < columns; k++)
				{
					temp_angle = k * angle;
					x = sin(temp_angle) * abs(radius * cos(temp_Theta));

					z = cos(temp_angle) * abs(radius * cos(temp_Theta));
					vertices.push_back(Coord(x, y, z));
				}

			}

			int DownPtNum = vertices.size();
			for (int i = 0; i < DownPtNum; i++)
			{
				if (vertices[i][1] != 0) { vertices.push_back(Coord(vertices[i][0], -vertices[i][1], vertices[i][2])); }
			}


			vertices.push_back(Coord(0, radius, 0));
			vertices.push_back(Coord(0, -radius, 0));
			
			int pt_side_len = vertices.size();




			//1.上半部分   （总点数/2）-1    
			//   i,i + 1,i + columns
			int fortriNum;
			if (row % 2 == 0) { fortriNum = (row - 4) / 2; }
			else { fortriNum = (row - 3) / 2; }
			for (int x = 0; x < fortriNum; x++)//(row - 2)
			{
				for (int i = 0; i < columns; i++)
				{
					int temp = x * columns;

					if ((i+1) % columns == 0)
					{
						triangle.push_back(TopologyModule::Triangle(i + temp, i - (columns-1)+temp, i - (columns - 1) +  columns +temp ));
						triangle.push_back(TopologyModule::Triangle(i - (columns - 1) + columns + temp,  i + columns + temp , i + temp));

					}
					else
					{
						triangle.push_back(TopologyModule::Triangle(i + temp, i + 1 + temp, i + columns + temp));
						triangle.push_back(TopologyModule::Triangle(i + 1 + temp, i + columns + 1 + temp, i + columns + temp));
					}
					//端面
					if (x == fortriNum - 1)
					{
						if ((i + 1) % columns != 0) 
						{
							triangle.push_back(TopologyModule::Triangle(i + temp + columns, i + temp + columns + 1, vertices.size() - 2));
						}
						else 
						{
							triangle.push_back(TopologyModule::Triangle(i + temp + columns, temp + columns, vertices.size() - 2));
						}
					}
				}

				//下半部分
				
				if (row % 2 == 0 | (row % 2 != 0 && x != fortriNum - 1))
				{
					for (int i = 0; i < columns; i++)
					{
						int temp = (x + fortriNum + 1) * columns;


						if ((i + 1) % columns == 0)
						{
							triangle.push_back(TopologyModule::Triangle(i - (columns - 1) + columns + temp, i - (columns - 1) + temp, i + temp));
							triangle.push_back(TopologyModule::Triangle(i + temp, i + columns + temp, i + 1  + temp));
							//std::cout << i + temp << "," << x * columns + temp << "," << (x + 1) * columns + temp << std::endl;

						}
						else
						{
							triangle.push_back(TopologyModule::Triangle(i + columns + temp, i + 1 + temp, i + temp));
							triangle.push_back(TopologyModule::Triangle(i + columns + temp, i + columns + 1 + temp, i + 1 + temp));
						}
						//端面
						if (x == fortriNum - 1)
						{
							if ((i + 1) % columns != 0)
							{
								triangle.push_back(TopologyModule::Triangle(vertices.size() - 1, i + temp + columns + 1, i + temp + columns));
							}
							else
							{
								triangle.push_back(TopologyModule::Triangle(vertices.size() - 1, temp + columns,i + temp + columns));
							}
						}
					}
				
				}
				else if (row % 2 != 0 && x == fortriNum - 1)
				{
					for (int i = 0; i < columns; i++)
					{
						int temp = (x + fortriNum) * columns;
						if (x == fortriNum - 1)
						{
							if ((i + 1) % columns != 0)
							{
								triangle.push_back(TopologyModule::Triangle(vertices.size() - 1, i + temp + columns + 1, i + temp + columns));
							}
							else
							{
								triangle.push_back(TopologyModule::Triangle(vertices.size() - 1, temp + columns, i + temp + columns));
							}
						}
					}
				}
				//中段连接处

					for (int i = 0; i < columns; i++)
					{
						int temp = (fortriNum + 1) * columns;
						if ((i + 1) % columns == 0) 
						{
							triangle.push_back(TopologyModule::Triangle(temp + i, i - columns + 1, i));
							triangle.push_back(TopologyModule::Triangle(temp + i - columns + 1, i - columns + 1, temp + i));
						}
						else 
						{
							triangle.push_back(TopologyModule::Triangle(temp + i, i + 1, i));
							triangle.push_back(TopologyModule::Triangle(temp + 1 + i, i + 1, temp + i));

						}
					}


			}



			//变换


			Quat<Real> q = Quat<Real>(M_PI * rot[0] / 180, Coord(1, 0, 0))
				* Quat<Real>(M_PI * rot[1] / 180, Coord(0, 1, 0))
				* Quat<Real>(M_PI * rot[2] / 180, Coord(0, 0, 1));

			q.normalize();

			auto RV = [&](const Coord& v)->Coord {
				return center + q.rotate(v - center);
			};

			int numpt = vertices.size();

			for (int i = 0; i < numpt; i++)
			{

				vertices[i] = RV(vertices[i] * scale + RV(center));
			}


		

		}
		triangleSet->setPoints(vertices);

		
		triangleSet->setTriangles(triangle);

		//triangleSet->update();

		vertices.clear();
		triangle.clear();
		//triangle.clear();
	}

	template<typename TDataType>
	void SphereModel<TDataType>::disableRender() {
		glModule->setVisible(false);
	};

	DEFINE_CLASS(SphereModel);
}