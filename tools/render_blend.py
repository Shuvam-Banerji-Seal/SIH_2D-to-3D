"""Headless Blender: render a preview of a drone3d .blend (camera above and behind the model, sun light).

    blender --background mesh.blend --python tools/render_blend.py -- out.png [width height]
"""

import math
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1 :]
out = args[0]
w, h = (int(args[1]), int(args[2])) if len(args) >= 3 else (1600, 900)
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
centre, size = (lo + hi) / 2, (hi - lo).length
cam = bpy.data.objects.new("preview_cam", bpy.data.cameras.new("preview_cam"))
bpy.context.scene.collection.objects.link(cam)
cam.location = centre + Vector((0.45, -0.9, 0.55)).normalized() * size * 0.95
direction = centre - cam.location
cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
cam.data.lens = 35
cam.data.clip_end = size * 10
bpy.context.scene.camera = cam
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.rotation_euler = (math.radians(40), 0, math.radians(30))
bpy.context.scene.collection.objects.link(sun)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items} else "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = w, h
scene.render.filepath = out
world = bpy.data.worlds.new("w") if scene.world is None else scene.world
scene.world = world
world.color = (0.05, 0.06, 0.08)
bpy.ops.render.render(write_still=True)
print("rendered", out, "z-range", round(lo.z, 2), round(hi.z, 2), "xy-extent", round(hi.x - lo.x, 2), round(hi.y - lo.y, 2))
