"""Headless Blender: import a GLB (y-up, as glTF requires) and save it as a .blend with its texture packed.

    blender --background --factory-startup --python tools/glb_to_blend.py -- in.glb out.blend

Blender's glTF importer turns y-up into Blender's z-up, so the survey frame
(east, north, up -- or the levelled SfM frame) comes back as it was.
"""

import sys

import bpy

src, dst = sys.argv[sys.argv.index("--") + 1 :][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        obj.name = "drone3d_model"
        for mat in obj.data.materials:  # the atlas is a photo: show it unlit-ish in the viewport
            if mat is not None:
                mat.use_backface_culling = False
bpy.ops.file.pack_all()
bpy.context.scene.unit_settings.system = "METRIC"
bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True)
