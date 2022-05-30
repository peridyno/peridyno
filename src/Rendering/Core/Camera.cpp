#include "Camera.h"

namespace dyno
{
	TRay3D<float> Camera::castRayInWorldSpace(float x, float y) 
	{
		float width = this->viewportWidth();
		float height = this->viewportHeight();

		glm::mat4 projMat = this->getProjMat();
		glm::mat4 viewMat = this->getViewMat();
		float dx = (2.0f * x) / width - 1.0f;
		float dy = 1.0f - (2.0f * y) / height;
		float dz = 0.95;

		glm::mat4 mv = glm::inverse(projMat*viewMat);

		glm::vec4 near = glm::vec4(dx, dy, -1, 1.0);
		glm::vec4 far = glm::vec4(dx, dy, 1, 1.0);

		glm::vec4 near_world = mv * near;
		glm::vec4 far_world = mv * far;
	
		if (near_world.w != 0.0)
		{
			near_world /= near_world.w;
		}

		if (far_world.w != 0.0)
		{
			far_world /= far_world.w;
		}

		TRay3D<float> ray;
		ray.origin = Vec3f(near_world.x, near_world.y, near_world.z);
		ray.direction = Vec3f(far_world.x - near_world.x, far_world.y - near_world.y, far_world.z - near_world.z).normalize();

		return ray;
	}
}
