mat4 build_rotate_align(vec3 a, vec3 b) {
  vec3 axis = cross(a, b);
  const float cos_a = dot(a, b);
  const float w = 1.0 + cos_a;
  float k = 0.0;
  // cos_a != -1
  if (w > 0.0001) {
    k = 1.0 / w;
  }

  // clang-format off
	return mat4(
		(axis.x * axis.x * k) + cos_a,  (axis.x * axis.y * k) - axis.z, (axis.z * axis.x * k) + axis.y, 0.0,
		(axis.x * axis.y * k) + axis.z, (axis.y * axis.y * k) + cos_a,  (axis.y * axis.z * k) - axis.x, 0.0,
		(axis.z * axis.x * k) - axis.y, (axis.y * axis.z * k) + axis.x, (axis.z * axis.z * k) + cos_a,  0.0,
		0.0,                            0.0,                            0.0,                            1.0
	);
  // clang-format on
}

mat4 build_translate(vec3 t) {
  // clang-format off
  return mat4(
    1,   0,   0,   0,
    0,   1,   0,   0,
    0,   0,   1,   0,
    t.x, t.y, t.z, 1
  );
  // clang-format on
}

mat4 build_scale(float s) {
  // clang-format off
  return mat4(
    s, 0, 0, 0,
    0, s, 0, 0,
    0, 0, s, 0,
    0, 0, 0, 1
  );
  // clang-format on
}