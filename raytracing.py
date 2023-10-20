# SADS A23 - Raytracer
# Code adapted from https://github.com/jamesbowman/raytrace, see LICENSE for info.
# Added triangle intersection and Stanford bunny loader.

from PIL import Image
import numpy as np

import bunny
import random

w = 400
h = 300

def normalize(x):
    x /= np.linalg.norm(x)
    return x

def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d

def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the 
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf

def intersect_triangle(O, D, a, b, c):
    # Test if the (O-D) ray intersects with the triangle formed by (a,b,c).
    EPS = 1e-6

    # Moller-Trumbore - https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html

    ab = b - a
    ac = c - a
    pv = np.cross(D, ac)
    det = np.dot(ab, pv)

    if (abs(det) < EPS):
        return np.inf # Ray parallel to the plane
    
    idet = 1 / det
    tv = O - a

    u = np.dot(tv, pv) * idet
    if (u < 0) | (u > 1):
        return np.inf
    
    qv = np.cross(tv, ab)
    v = np.dot(D, qv) * idet

    if (v < 0) | ((u+v) > 1):
        return np.inf

    t = np.dot(ac, qv) * idet
    return t

def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])
    elif obj['type'] == 'triangle':
        return intersect_triangle(O, D, obj['p1'], obj['p2'], obj['p3'])

def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'triangle':
        return np.array([0,0,-1]) # TODO!
    return N
    
def get_color(obj, M):
    color = obj['color']
    if not hasattr(color, '__len__'):
        color = color(M)
    return color

def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Shadow: find if the point is shadowed or not.
    l = [intersect(M + N * .0001, toL, obj_sh) 
            for k, obj_sh in enumerate(scene) if k != obj_idx]
    if l and min(l) < np.inf:
        return
    # Start computing the color.
    col_ray = ambient
    # Lambert shading (diffuse).
    col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col_ray += obj.get('specular_c', specular_c) * max(np.dot(N, normalize(toL + toO)), 0) ** specular_k * color_light
    return obj, M, N, col_ray

def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position), 
        radius=np.array(radius), color=np.array(color), reflection=.5)
    
def add_plane(position, normal):
    return dict(type='plane', position=np.array(position), 
        normal=np.array(normal),
        color=lambda M: (color_plane0 
            if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1),
        diffuse_c=.75, specular_c=.5, reflection=.25)

def add_triangle(p1, p2, p3, color):
    return dict(type="triangle", p1=np.array(p1), p2=np.array(p2), p3=np.array(p3),
                color=np.array(color), reflection=.5)
# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)
scene = [#add_sphere([.75, .1, 1.], .6, [0., 0., 1.]),
      #   add_sphere([-.75, .1, 2.25], .6, [.5, .223, .5]),
      #   add_sphere([-2.75, .1, 3.5], .6, [1., .572, .184]),
      #   add_triangle([-1,0.4,2], [1,0.4,2], [0,0.75,2], [.8,.8,.8]),
         add_plane([0., -.5, 0.], [0., 1., 0.]),
    ]

# Sample cube:
#vertices = [
#    (-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1),
#    (-1, 2, -1), (1, 2, -1), (1, 2, 1), (-1, 2, 1)
#]
#indices = [
#    (0, 1, 2), (2, 3, 0),       # Bottom
#    (0, 1, 5), (5, 4, 0),       # Front
#    (4, 5, 6), (6, 7, 4),       # Top
#    (1, 2, 6), (6, 5, 1),       # Right
#    (0, 4, 7), (7, 3, 0),       # Left
#    (2, 3, 7), (7, 6, 2),       # Back
#
#]

# Adding the Stanford bunny:
vertices, indices = bunny.load("bun_zipper_res4.ply")

for i in random.sample(range(0, len(indices)), len(indices) / 4): # Random sample of 25% of the triangle
#for i in range(0,len(indices)):
    p1 = np.array(vertices[indices[i][0]])
    p2 = np.array(vertices[indices[i][1]])
    p3 = np.array(vertices[indices[i][2]])
    scene.append(
        add_triangle(p1, p2, p3, [.8, .8, .8])
    )

# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)

# Default light and material parameters.
ambient = .05
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 1 # Maximum number of light reflections.
col = np.zeros(3)  # Current color.
O = np.array([0., 0.35, -3.])  # Camera.
Q = O + np.array([0, 0, 1]) # Camera pointing to (absolute coords).
img = np.zeros((h, w, 3))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

# Loop through all pixels.
for i, x in enumerate(np.linspace(S[0], S[2], w)):
    if i % 10 == 0:
        print(i / float(w) * 100, "%")

    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        Q[:2] = (x, y)
        D = normalize(Q - O)

        depth = 0
        rayO, rayD = O, D
        reflection = 1.
        # Loop through initial and secondary rays.
        while depth < depth_max:
            traced = trace_ray(rayO, rayD)
            if not traced:
                break
            obj, M, N, col_ray = traced
            # Reflection: create a new ray.
            rayO, rayD = M + N * .0001, normalize(rayD - 2 * np.dot(rayD, N) * N)
            depth += 1
            col += reflection * col_ray
            reflection *= obj.get('reflection', 1.)
        img[h - j - 1, i, :] = np.clip(col, 0, 1)

im = Image.fromarray((255 * img).astype(np.uint8), "RGB")
im.save("fig.png")
