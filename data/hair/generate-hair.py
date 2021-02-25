import math

min_radius = 0.3
max_radius = 0.4
seg_radius = 10

avg_dis = 0.01

min_phi = math.pi/4
max_phi = math.pi/2
seg_phi = int((max_phi-min_phi)*min_radius/avg_dis)

vert_list = []
edge_list = []
for i in range(seg_phi+1):
    phi = min_phi+(max_phi-min_phi)/seg_phi*i
    peri = 2*math.pi*min_radius*math.cos(phi)
    seg_theta = int(peri/avg_dis)
    for j in range(seg_theta+1):
        if seg_theta == 0:
            seg_theta = 1 
        theta = 2*math.pi/seg_theta*j
        for k in range(seg_radius+1):
            radius = min_radius+(max_radius-min_radius)/seg_radius*k
            vert_list.append([radius*math.cos(phi)*math.cos(theta),radius*math.cos(phi)*math.sin(theta),radius*math.sin(phi)])
        vert_size = len(vert_list)
        for k in range(seg_radius):
            edge_list.append([vert_size-k, vert_size-k-1])

with open('hair.obj','w') as f:
    for i in range(len(vert_list)):
          f.write('v %f %f %f\n' % (vert_list[i][0], vert_list[i][1], vert_list[i][2]))

with open('hair.smesh', 'w') as f:
    f.write('*VERTICES\n')
    f.write('%d %d\n' % (len(vert_list),3))
    for i in range(len(vert_list)):
        f.write('%d %f %f %f\n' % (i+1, vert_list[i][0], vert_list[i][1], vert_list[i][2]))
    f.write('*ELEMENTS\n')
    f.write('LINE\n')
    f.write('%d %d\n' % (len(edge_list), 2))
    for i in range(len(edge_list)):
        f.write('%d %d %d\n' % (i+1, edge_list[i][0], edge_list[i][1]))    