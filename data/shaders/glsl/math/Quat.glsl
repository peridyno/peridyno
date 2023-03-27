#ifndef QUAT_H
#define QUAT_H

vec4 quat_from_axis_angle(vec3 axis, float angle)
{ 
  vec4 qr;
  float half_angle = (angle * 0.5f) * radians(180.0f) / 180.0f;
  qr.x = axis.x * sin(half_angle);
  qr.y = axis.y * sin(half_angle);
  qr.z = axis.z * sin(half_angle);
  qr.w = cos(half_angle);
  return qr;
}

vec4 quat_from_axis_angle_rad(vec3 axis, float angle)
{
  float half_angle = angle * 0.5f;
  return vec4(axis * sin(half_angle), cos(half_angle));
}

vec3 quat2eulZYX(vec4 q)
{
  q = normalize(q);
  q = q.w >= 0 ? q : -q;
  float x = q.x, y = q.y, z = q.z, w = q.w;
  return vec3( // in xyz
    atan(2.0f * (y*z + w*x), w*w - x*x - y*y + z*z),
    asin(-2.0f * (x*z - w*y)),
    atan(2.0f * (x*y + w*z), w*w + x*x - y*y - z*z)
  );
}

vec4 eul2quatZYX(vec3 v)
{
  float cy = cos(v.z * 0.5f);
  float sy = sin(v.z * 0.5f);
  float cp = cos(v.y * 0.5f);
  float sp = sin(v.y * 0.5f);
  float cr = cos(v.x * 0.5f);
  float sr = sin(v.x * 0.5f);
  return vec4(
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
    cr * cp * cy + sr * sp * sy
  );
}

vec4 quat_conj(vec4 q)
{ 
  return vec4(-q.x, -q.y, -q.z, q.w); 
}

float quat_norm_squared(vec4 q)
{ 
  return q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w; 
}

vec4 quat_inverse(vec4 q)
{ 
  return quat_conj(q)/quat_norm_squared(q);
}
  
vec4 quat_mul(vec4 q1, vec4 q2)
{ 
  vec4 qr;
  qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
  qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
  qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
  qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
  return qr;
}

vec3 rotate_vertex_position(vec3 position, vec3 axis, float angle)
{ 
  vec4 qr = quat_from_axis_angle(axis, angle);
  vec4 qr_conj = quat_conj(qr);
  vec4 q_pos = vec4(position.x, position.y, position.z, 0);
  
  vec4 q_tmp = quat_mul(qr, q_pos);
  qr = quat_mul(q_tmp, qr_conj);
  
  return vec3(qr.x, qr.y, qr.z);
}

vec4 quat_normalize(vec4 quat)
{
  return normalize(quat);
}

vec3 quat_rotate(vec4 quat, vec3 v)
{
  // Extract the vector part of the quaternion
  vec3 u = quat.xyz;

  // Extract the scalar part of the quaternion
  float s = quat.w;

  // Do the math
  return    2.0f * dot(u, v) * u
      + (s*s - dot(u, u)) * v
      + 2.0 * s * cross(u, v);
}

mat3 quat_to_mat3(vec4 quat)
{
  float x2 = quat.x + quat.x;
  float y2 = quat.y + quat.y;
  float z2 = quat.z + quat.z;
  float xx = x2 * quat.x;
  float yy = y2 * quat.y;
  float zz = z2 * quat.z;
  float xy = x2 * quat.y;
  float xz = x2 * quat.z;
  float xw = x2 * quat.w;
  float yz = y2 * quat.z, yw = y2 * quat.w, zw = z2 * quat.w;
  //note opengl store matrix in column major order
  return transpose(mat3(1.0 - yy - zz, xy - zw, xz + yw,
              xy + zw, 1.0 - xx - zz, yz - xw,
              xz - yw, yz + xw, 1.0 - xx - yy));
}

#endif